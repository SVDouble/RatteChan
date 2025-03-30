# Ideas
- dynamic target deflection based on spline curvature
- sensor is at an angle with the surface (i.e. the whisker is looking back)

# Experience
- target proportional to curvature is bad as slippage is likely to occur
  ( should not allow sudden deflection changes )
- body predictions (tip - offset(defl)) are not precise

# Viewer Control
See https://github.com/google-deepmind/mujoco/blob/main/doc/python.rst


# Sensors
datasheet: https://cdn-learn.adafruit.com/assets/assets/000/069/600/original/MLX90393-Datasheet-Melexis.pdf?1547824268
jst 4-pin connector: https://www.adafruit.com/product/4399

### Env.SENSORS
```json
[{"name":"L0","bus":1,"address":"0x1a"},{"name":"L1","bus":1,"address":"0x18"},{"name":"L2","bus":2,"address":"0x19"},{"name":"R0","bus":1,"address":"0x1b"},{"name":"R1","bus":1,"address":"0x19"},{"name":"R2","bus":2,"address":"0x18"}]
```
