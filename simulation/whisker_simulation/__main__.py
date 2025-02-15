from whisker_simulation.simulation import WhiskerSimulation
from whisker_simulation.utils import get_config

config = get_config()
simulation = WhiskerSimulation(config)
simulation.run()
