# Vendored UR10 CB3 visual model

Source repository: <https://github.com/rudra-8000/ur10-mujoco-sim/tree/main/ur10sim>

Imported from upstream commit `7291ed3b8e541a7e5788e1b4b943deef5f470580`:

- `ur10.xml`
- `meshes/*.stl`
- `ur10_meta.json`
- the source parameter YAML files
- `LICENSE_UniversalRobots_BSD3`

The source mesh/model parameters originate from the Universal Robots ROS 2
description at commit `89bbe795f38a7ab00fb66fe8831dfff79dc99edf`. The copied
license is retained alongside the model.

The integration boundary is the arm body and its visual meshes only. The
project's existing end-effector assembly is retained: the fixed-closed
Robotiq 2F-85, brush handle and plate/contact geometry, wrist camera, force /
torque and TCP sensors, task scene, ACT actions, and admittance controller.
The gripper is the same fixed tool used by the existing simulation; it is not
an actively opening/closing gripper.

The upstream MJCF has six bare joints but no arm actuators, so this profile
adapts the joint-name mapping and adds position servos for those arm joints;
this does not replace the retained end-effector/tool assembly. The source UR
visual geoms have contact disabled, and the generated per-link inertias are
simplified. Treat this profile as a visual/kinematic compatibility experiment,
not as a validated high-fidelity collision or dynamics model.
