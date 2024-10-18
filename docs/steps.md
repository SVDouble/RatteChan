# MVP

3. Build a single whisker
   - 3D print the base (2 parts) and suspension
   - cut the nitinol wire and glue the magnet to the tip
   - glue the base together
   - assemble the sensor

2. Wire the single whisker
   - connect the hall effect sensor to the Arduino
   - write code to quickly get basic measurements
   - put the hall sensor into the whisker base and try to calibrate it
   - get the first true measurement results

3. Assemble the improvised whisker array
   - connect all 3 single whiskers to the Arduino
   - write code to get measurements from all 3 whiskers
   - write code to visualize the measurements
   - develop a algorithm to combine the measurements and output 3 3D points in one coordinate system

4. Basic whisker array testing
   - sweep a certain profile and try to generate a surface
   - real time data visualization
   - try to detect a simple object

5. Advanced whisker array assembly
   - develop a new whisker base that can hold multiple whiskers
   - develop the wiring system that can connect all whiskers to the Arduino
   - assemble the whisker array
   - test it to construct a 3D surface
   - try to detect a more complex object

## Progress so far

### 17.10.2024 (15:30-16:30)
- [x] register for the bachelor thesis (awaiting approval from the Studienbüro)
- [x] export suspension, base and top to STL
- [x] create a proper gcode for all parts

## TODO

### 18.10.2024 (12:30-16:30)
- [ ] 3D print all the parts (should take 20 mins unless goes wrong)
- [ ] solder together arduino and hall effect sensor
- [ ] run the test code for the hall effect sensor
- [ ] cut the whisker wire and glue the magnet to the tip
- [ ] get first measurements from the whisker (console output)

## 19.10.2024 (08:00-11:00)
- [ ] write the algorithm to properly process measurements from the whisker (2D coordinate plot)
- [ ] write measurements to Redis, read them from a different routine
- [x] check whether grafana + websocket can be used for realtime signal visualization
  - if not, check out tools like vizpy and other opengl-based libraries (although grafana is better imho)

## 21.10.2024 (10:00-18:00)
- [ ] print and assemble 2 more whiskers
- [ ] connect all 3 whiskers to the Arduino
- [ ] write code to get measurements from all 3 whiskers
- [ ] display all the measurements at the same time

## 24.10.2024 (10:00-16:30)
- [ ] try out different configurations of the base
- [ ] find out about algorithms to combine the measurements, maybe generate a surface
- TBD