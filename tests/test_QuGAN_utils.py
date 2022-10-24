from unittest import TestCase
from QuGAN import *
from qiskit import Aer

D, D_params = create_QuGAN_ansatz(n_qubits=3, n_layer=4, param_label='D', register_label='d', out_reg_index=2)
G, G_params = create_QuGAN_ansatz(n_qubits=2, n_layer=2, param_label='G', register_label='g', out_reg_index=0)
R = QuantumCircuit(2)
R.cx(1, 0)
grad_storage = get_gradient_circuit_storage(D, G, R)
q_inst = QuantumInstance(Aer.get_backend('statevector_simulator'))

class Test(TestCase):
    def test_get_expectation_values(self):
        return_list = Test().test_allocate_parameters()
        print('Got return list')
        results = get_expectation_values(return_list, q_inst)
        print(results)

    def test_allocate_parameters(self):
        grad_D_with_G_circs = grad_storage[0][1]
        return_list = allocate_parameters(grad_D_with_G_circs, param_frame_G=G_params,
                                            param_values_G=range(len(G_params)),
                                            param_frame_D=D_params, param_values_D=range(len(D_params)))
        return return_list
