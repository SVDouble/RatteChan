from whisker_simulation.simulation import WhiskerSimulation

simulation = WhiskerSimulation(
    model_path="models/whisker.xml",
    duration=160,
    camera_fps=30,
    control_rps=100,
)
simulation.run()
