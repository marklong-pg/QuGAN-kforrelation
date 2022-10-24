from qiskit import QuantumCircuit, QuantumRegister, Aer
from qiskit.circuit import ParameterVector
from qiskit.opflow import Z, I, CircuitSampler, StateFn, AerPauliExpectation
from qiskit.opflow import OperatorBase
from qiskit.utils import QuantumInstance
from functools import partial
import multiprocessing as mp
from multiprocessing import set_start_method
import numpy as np
import os
os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'

# TODO: be more fancy and use stream/functional programming for some of the iterative operations

def train_bin_conditioned_QuGAN(R, n_qubits_G, n_layer_G, n_qubits_D, n_layer_D, q_instance, n_steps=10000, n_train_G=100,
                initial_rate=10, balance_rate=0.1, retain_factor=800, const_threshold=None):
    """
    Train QuGAN
    :param q_instance: QuantumInstance
    :param R: QuantumCircuit, real source in form of quantum circuit
            requires: R has the same Out, Label, Bath registers as G
            # TODO: loosen this requirement
    :param n_qubits_G: int,
    :param n_layer_G: int,
    :param n_qubits_D: int,
    :param n_layer_D: int,
    :param n_steps: int, number of gradient steps from training
    :param n_train_G: int: spacing of steps between each consecutive training of G
    :param initial_rate: int,
    :param balance_rate: int,
    :param retain_factor: int, value for learning rate retaintion, higher retain_factor, slower decay of learning rate
                recommend: retain_factor in [500,1000]
    :param const_threshold: float, number of steps with decaying learning rate
    :return:
    """

    V_DR = []
    V_DG = []
    V = []
    labels = [0,1]
    D, D_pr_frame = create_QuGAN_ansatz(n_qubits=n_qubits_D, n_layer=n_layer_D, param_label='D',
                                      register_label='d', out_reg_index=-1)
    G, G_pr_frame = create_QuGAN_ansatz(n_qubits=n_qubits_G, n_layer=n_layer_G, param_label='G',
                                      register_label='g', out_reg_index=0)
    D_pr_vals = 2*np.random.rand(len(D_pr_frame))-1
    G_pr_vals = 2*np.random.rand(len(G_pr_frame))-1
    gradient_circ_storage = get_gradient_circuit_storage(D,G,R)
    circ_sampler = CircuitSampler(q_instance,param_qobj=True)
    grad_exp_meas = AerPauliExpectation().convert(~StateFn((I ^ (gradient_circ_storage[0][0][0].num_qubits - 1)) ^ Z))

    # get learning rate vector
    learning_rates = list(map(lambda x: (initial_rate-balance_rate)*np.exp(-x/retain_factor)+balance_rate, range(n_steps)))
    if const_threshold is not None:
        learning_rates = learning_rates[:const_threshold] + [balance_rate,]*(n_steps-const_threshold)

    for step in range(n_steps):
        print(f'Step: {step}, learning rate: {learning_rates[step]}...')
        grad_D = np.zeros(len(D_pr_frame))
        for label in labels:
            grad_circ_DR = gradient_circ_storage[label][0]
            grad_circ_DG = gradient_circ_storage[label][1]
            assigned_circuits_DR = allocate_parameters(grad_circ_DR,param_frame_D=D_pr_frame,param_values_D=D_pr_vals)
            assigned_circuits_DG = allocate_parameters(grad_circ_DG,param_frame_D=D_pr_frame,param_values_D=D_pr_vals,
                                                       param_frame_G=G_pr_frame, param_values_G=G_pr_vals)
            grad_D = grad_D + 1/(4*len(labels)) * (get_expectation_values(assigned_circuits_DR,grad_exp_meas,circ_sampler) +
                                                   get_expectation_values(assigned_circuits_DG,grad_exp_meas,circ_sampler))
        D_pr_vals = D_pr_vals + learning_rates[step] * grad_D
        if step % n_train_G == 0:
            grad_G = np.zeros(len(G_pr_frame))
            for label in labels:
                grad_circ_G = gradient_circ_storage[label][2]
                assigned_circuits_G = allocate_parameters(grad_circ_G,param_frame_D=D_pr_frame,param_values_D=D_pr_vals,
                                                          param_frame_G=G_pr_frame, param_values_G=G_pr_vals)
                grad_G = grad_G + 1/(4*len(labels)) * (get_expectation_values(assigned_circuits_G,grad_exp_meas,circ_sampler))
            G_pr_vals = G_pr_vals + 5*learning_rates[step] * grad_G

        if step % 50 == 0:
            v_dr, v_dg, v = get_performance(D,G,R,D_pr_frame,D_pr_vals,G_pr_frame,G_pr_vals,circ_sampler)
            for (x,y) in [(V_DR,v_dr),(V_DG,v_dg),(V,v)]: x.append(y)
            np.savetxt('results/V_DR.txt', V_DR, fmt='%.8f')
            np.savetxt('results/V_DG.txt', V_DG, fmt='%.8f')
            np.savetxt('results/V.txt', V, fmt='%.8f')
    np.savetxt('results/G_pr_vals.txt',G_pr_vals)
    np.savetxt('results/D_pr_vals.txt',D_pr_vals)
    return (G,G_pr_frame,G_pr_vals), (D,D_pr_frame,D_pr_vals)

def get_performance(D, G, R, D_pr_frame, D_pr_vals, G_pr_frame, G_pr_vals, circ_sampler):
    """Get cost values of QuGAN with given parameters"""
    RD_0, GD_0 = get_performance_circuits(D,G,R,0)
    RD_1, GD_1 = get_performance_circuits(D,G,R,1)
    ordered_circ_list = [RD_0, RD_1, GD_0, GD_1]
    ordered_circ_list = allocate_parameters(ordered_circ_list,D_pr_frame,D_pr_vals,G_pr_frame,G_pr_vals,is_grad=False)
    exp_meas = AerPauliExpectation().convert(~StateFn((I ^ (RD_0.num_qubits - 1))^Z))
    exp_vals = get_expectation_values(ordered_circ_list,exp_meas,circ_sampler)
    V_DR = (1/8) * (exp_vals[0] + exp_vals[1])
    V_DG = -(1/8) * (exp_vals[2] + exp_vals[3])
    V = 1/2 + V_DR + V_DR
    return V_DR, V_DG, V

def create_QuGAN_ansatz(n_qubits: int, n_layer: int, register_label='q', out_reg_index = None,
                        param_label=''r'$\theta$', **kwargs):
    """
    Create QuGAN ansatz of n_layer
    :param out_reg_index: int, index of output register
    :param register_label: str, label for the quantum register
    :param n_qubits: int, number of qubits in the ansatz
    :param n_layer: int, number of layers for the ansatz
    :param param_label: str, label for the parameters
    :return: ansatz: QuantumCircuit, QuGAN ansatz of n_layer
    """
    # TODO: implement correct register name for label (e.g., label, bath, out)
    qr = list(QuantumRegister(n_qubits,register_label))
    qr[out_reg_index] = QuantumRegister(1,'out')[0]
    ansatz = QuantumCircuit(qr)
    n_params_per_layer = 3*n_qubits-1
    params = ParameterVector(param_label, n_layer * n_params_per_layer)

    # initialization of initial state (use for label conditioning)
    for key, value in kwargs.items():
        ansatz.initialize(value,int(key))
    if kwargs: ansatz.barrier()

    for layer in range(n_layer):
        # add all the Rx and Rz gates
        base = n_params_per_layer * layer
        for qubit in range(n_qubits):
            ansatz.rx(params[base+qubit],qubit)
        for qubit in range(n_qubits):
            ansatz.rz(params[base+n_qubits+qubit],qubit)
        # add all the Rzz gates
        base = base + 2*n_qubits
        for qubit in np.arange(0,n_qubits-1,2):
            if qubit + 1 < n_qubits:
                ansatz.rzz(params[int(base+qubit/2)],qubit,qubit+1)
        base = base + int(n_qubits / 2)
        for qubit in np.arange(1,n_qubits-1,2):
            if qubit + 1 < n_qubits:
                ansatz.rzz(params[int(base+(qubit-1)/2)],qubit,qubit+1)
        # if layer < (n_layer - 1): ansatz.barrier()
    return ansatz, params

def get_gradient_circuit_storage(D:QuantumCircuit, G:QuantumCircuit, R:QuantumCircuit):
    """
    Create storage of all gradient templates
    :param D: QuantumCircuit, Discriminator ansatz
    :param G: QuantumCircuit, Generator ansatz
    :param R: QuantumCircuit, Real source ansatz
    :return: tuple(list(list(QuantumCircuit))), storage for gradient circuits of labels 0 and 1
            return[x][y][z]
            x : 0 or 1, conditional label
            y : 0 --> gradient circuits for D with R
                1 --> gradient circuits for D with G
                2 --> gradient circuits for G
            z : index of the target parameter
    """
    DR0 = []; DR1 = []
    DG0 = []; DG1 = []
    G0 = []; G1 =[]
    conditioned_frame = QuantumCircuit(QuantumRegister(1, 'grad'), D.qubits, G.qubits[1:])
    # conditioned_frame.initialize(1, G.qubits[1])
    # conditioned_frame.initialize(1, D.qubits[1])  # NOTE: will need to be changed later to QuantumRegister(label)
    conditioned_frame.x(G.qubits[1])
    conditioned_frame.x(D.qubits[1])
    for i in range(D.num_parameters):
        grad_DR = create_gradient_circuit(D,G,R,i,'D',with_G=False)
        grad_DG = create_gradient_circuit(D,G,R,i,'D',with_G=True)
        DR0.append(grad_DR)
        DG0.append(grad_DG)
        DR1.append(conditioned_frame.compose(grad_DR,range(conditioned_frame.num_qubits)))
        DG1.append(conditioned_frame.compose(grad_DG, range(conditioned_frame.num_qubits)))
    for i in range(G.num_parameters):
        grad_G = create_gradient_circuit(D,G,R,i,'G')
        G0.append(grad_G)
        G1.append(conditioned_frame.compose(grad_G,range(conditioned_frame.num_qubits)))
    return [DR0, DG0, G0], [DR1, DG1, G1]

def create_gradient_circuit(D:QuantumCircuit, G:QuantumCircuit, R:QuantumCircuit,target_param_index,
                            target_actor='D',with_G=True):
    """
    Get circuit to calculate gradient w.r.t. a target parameter
    :param target_actor: str, 'D' or 'G', where the target parameter belong to
    :param D: QuantumCircuit, Discriminator ansatz
    :param G: QuantumCircuit, Generator ansatz
    :param R: QuantumCircuit, Real source ansatz
    :param target_param_index: int, index of target parameter for gradient
    :param with_G: boolean, whether provider is Real or Generator
    :return: gradient_circuit: QuantumCircuit, quantum circuit for calculating the gradient w.r.t. target_param
    """
    grad_reg = QuantumRegister(1,'grad')
    grad_circ = QuantumCircuit(grad_reg,D.qubits,G.qubits[1:])
    grad_circ.h(0)
    czz = QuantumCircuit(2)
    czz.cz(0,1)
    czz_instruction = czz.to_gate().control(1)

    def attach_gradient_section(A):
        for i in range(len(A.data)):
            data = A.data[i] # ---> tuple: (instruction, qargs, cargs)
            if i != target_param_index: grad_circ.append(data[0], data[1])
            # else: grad_circ.append(instruction_to_controlled(data[0]), [grad_reg[0]] + data[1])
            else:
                if data[0].name == 'rx': grad_circ.cx(grad_reg,data[1])
                elif data[0].name == 'rz': grad_circ.cz(grad_reg,data[1])
                else: grad_circ.append(czz_instruction, [grad_reg[0]] + data[1])

    if target_actor == 'D':
        grad_circ = grad_circ.compose(G, G.qubits) if with_G else grad_circ.compose(R, range(-R.num_qubits, 0, 1))
        grad_circ.barrier()
        attach_gradient_section(D)
        grad_circ.barrier()
        if with_G: grad_circ.x(D.qubits[0])
    else:
        attach_gradient_section(G)
        grad_circ.barrier()
        grad_circ.compose(D, D.qubits, inplace=True)
        grad_circ.barrier()

    grad_circ.cz(grad_reg,D.qubits[0])
    grad_circ.h(grad_reg)
    grad_circ.rx(np.pi/2,grad_reg)
    return grad_circ

def get_performance_circuits(D: QuantumCircuit, G: QuantumCircuit, R: QuantumCircuit, label=0):
    """
    Get combined circuits of RD and GD
    :param D: QuantumCircuit, Discriminator ansatz
    :param G: QuantumCircuit, Generator ansatz
    :param R: QuantumCircuit, Real source ansatz
    :param label: int, label 0 or 1
    :return: RD and GD ansatz for QuGAN
    """
    total_n_qubits = G.num_qubits + D.num_qubits - 1
    circuit_frame = QuantumCircuit(total_n_qubits)
    if label == 1:
        circuit_frame.x([1,3]) # TODO: change/remove the magic numbers
    RD_ansatz = circuit_frame.compose(R, range(-R.num_qubits, 0, 1))
    RD_ansatz.barrier()
    RD_ansatz.compose(D, range(D.num_qubits),inplace=True)

    GD_ansatz = circuit_frame.compose(G, range(-G.num_qubits, 0, 1))
    GD_ansatz.barrier()
    GD_ansatz.compose(D, range(D.num_qubits),inplace=True)
    return RD_ansatz, GD_ansatz

def instruction_to_controlled(instruction):
    """
    Creat controlled operation from a given instruction
    :param instruction:
    :return:controlled instruction
    """
    qc = QuantumCircuit(instruction.num_qubits)
    qc.append(instruction,range(instruction.num_qubits))
    return qc.to_gate().control(1)

# TODO: parallelize the operations
def get_expectation_values(list_circuits:list[QuantumCircuit],exp_meas:OperatorBase, circ_sampler:CircuitSampler):
    """
    Get expectation value for Z measurement at the first qubit for all circuits
    :param list_circuits: list[QuantumCircuit], list of all quantum circuits for measurement
    :param exp_meas: OperatorBase, expectation operator for the circuit
    :param circ_sampler: CircuitSampler, to sample the circuit
    :return: list[float]: list of expectation value, one for each circuit
    """
    # pool = mp.Pool(2)
    # func_handle = partial(cal_expectation,circ_sampler=circ_sampler,exp_meas=exp_meas)
    # results = pool.map(func_handle, list_circuits)
    # pool.close()
    results = []
    for qc in list_circuits:
        results.append(circ_sampler.convert(exp_meas @ StateFn(qc)).eval().real)
    return np.array(results)

# def cal_expectation(circ_sampler:CircuitSampler, exp_meas, qc:QuantumCircuit):
#     return circ_sampler.convert(exp_meas @ StateFn(qc)).eval().real

def allocate_parameters(list_circuits, param_frame_D : ParameterVector, param_values_D,
                        param_frame_G : ParameterVector = None, param_values_G=None, is_grad=True):
    """
    Assign G and D parameters to a list of circuits
    :param list_circuits: (list)[QuantumCircuit], circuit(s) to assign parameters
    :param param_frame_G: ParameterVector, params view for G
    :param param_values_G: list[float], list of parameters values to assign to G
    :param param_frame_D: ParameterVector, params view for D
    :param param_values_D: list[float], list of parameters values to assign to D
    :param is_grad: boolean, whether list of circuits are gradient circuits
    :return: (list)[QuantumCircuit], circuit(s) with parameters assigned
    """
    if type(list_circuits) is QuantumCircuit:
        list_circuits = [list_circuits]
    if param_frame_G is None or param_values_G is None:
        param_frame_G = param_values_G = []
    return_list_circuits = []
    params_dict_D = {param: value for (param, value) in zip(param_frame_D, param_values_D)}
    params_dict_G = {param: value for (param, value) in zip(param_frame_G, param_values_G)}
    if is_grad:
        params_dict_G.update(params_dict_D)
        params_dict = params_dict_G
        for qc in list_circuits:
            missing_param = find_not_in_circuit(qc,list(params_dict.keys()))
            temp_dict = params_dict.copy()
            temp_dict.pop(missing_param)
            return_list_circuits.append(qc.bind_parameters(temp_dict))
    else:
        for qc in list_circuits:
            qc = qc.bind_parameters(params_dict_D)
            if len(param_values_G) != 0 and param_frame_G[0] in qc.parameters:
                qc = qc.bind_parameters(params_dict_G)
            return_list_circuits.append(qc)
    # pool = mp.Pool(4)
    # def attach_all(list_circuits_, params_dict_):
    #     return_list = []
    #     for qc in list_circuits_:
    #         return_list.append(qc.bind_parameters(params_dict_))
    #     return return_list
    # pool.close()
    return return_list_circuits

def find_not_in_circuit(circuit:QuantumCircuit, params_vec):
    """Find which param in params is not present in the circuit"""
    return_param = None
    for param in params_vec:
        if param not in circuit.parameters:
            return_param = param
            break
    return return_param

# def attach_params(qc:QuantumCircuit,params_dict_):
#     return qc.bind_parameters(params_dict_)

# === TO RUN ===
if __name__ == "__main__":
    q_inst = QuantumInstance(backend= Aer.get_backend('statevector_simulator'))
    R = QuantumCircuit(2)
    R.cx(1, 0)
    train_bin_conditioned_QuGAN(R=R, n_qubits_G=2, n_layer_G=2, n_qubits_D=3, n_layer_D=4,
                                q_instance=q_inst, n_steps=10000,  n_train_G=100, initial_rate=10, balance_rate=0.1,
                                retain_factor=800, const_threshold=4000)