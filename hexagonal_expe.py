from slave_rotor_hexagonal_class import honeycomb
import matplotlib.pyplot as plt


Hubbard = honeycomb({"t":1.})
[Qf,Z] = Hubbard.solve_self_consistent_equation(7.0,[0.1891316, 2.09576934],'mott')

#U = 3.5 , x0 = [1.091319, 1.5757]
#U = 4.0, x0 = [0.1091319, 1.6957]