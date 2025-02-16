from whisker_simulation.simulation import Simulation
from whisker_simulation.utils import get_config

config = get_config()
simulation = Simulation(config)
simulation.run()
