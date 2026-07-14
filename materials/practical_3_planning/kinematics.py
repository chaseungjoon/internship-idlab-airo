import math
from typing import List, Union

import numpy as np
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from loguru import logger
from pydrake.geometry import Cylinder, Rgba
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody import inverse_kinematics
from pydrake.multibody.tree import Frame, ModelInstanceIndex
from pydrake.planning import RobotDiagram
from pydrake.solvers import IpoptSolver, SnoptSolver


def add_meshcat_triad(
    meshcat, path, length=0.05, radius=0.002, opacity=1.0, X_W_Triad=RigidTransform(), rgba_xyz=None
):
    if rgba_xyz is None:
        rgba_xyz = [[1, 0, 0, opacity], [0, 1, 0, opacity], [0, 0, 1, opacity]]

    meshcat.SetTransform(path, X_W_Triad)
    # x-axis
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0])
    meshcat.SetTransform(path + "/x-axis", X_TG)
    meshcat.SetObject(path + "/x-axis", Cylinder(radius, length), Rgba(*rgba_xyz[0]))

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0])
    meshcat.SetTransform(path + "/y-axis", X_TG)
    meshcat.SetObject(path + "/y-axis", Cylinder(radius, length), Rgba(*rgba_xyz[1]))

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    meshcat.SetTransform(path + "/z-axis", X_TG)
    meshcat.SetObject(path + "/z-axis", Cylinder(radius, length), Rgba(*rgba_xyz[2]))


class RobotKinematics:
    ur_plausible_configurations = [
        # TODO: these are solutions hardcoded for open_cover_controller, but we actually want the first joint to rotate in the other direction so do it +2pi
        [
            -2.6190064698820663,
            -2.5077217202117055,
            -0.3827338472277641,
            -4.895062536208747,
            -1.5271704550471643,
            1.2268092212143764,
        ],  # open_cover_controller good starting solution, however close to singularity
        [
            0.49433767795562744,
            -2.339724203149313,
            1.6764219442950647,
            -2.620105882684225,
            -0.8053491751300257,
            3.16013240814209,
        ],  # close to current home q
        [0, 0, 0, 0, 0, 0],
        [0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0],
        [0, -math.pi / 2, 0, -math.pi / 2, -math.pi / 2, math.pi / 2],
        [
            3.6237240490960114,
            -2.4507657004095766,
            -0.48247947564376226,
            -4.841883325429231,
            4.676157905141756,
            1.2119831862377273,
        ],  # open_cover_controller good starting solution, however close to singularity
    ]

    default_position_tolerance = [[0.001, 0.001, 0.001], [0.001, 0.001, 0.001]]
    default_rotation_tolerance = math.radians(1)

    def __init__(
        self,
        robot_diagram: RobotDiagram,
        arm_model_index: ModelInstanceIndex,
        meshcat=None,
    ):
        self.robot_diagram = robot_diagram
        self.plant = robot_diagram.plant()

        self.arm_model_index = arm_model_index

        self.tcp_frame_name = "tcp"
        self.tcp_frame = self.plant.GetFrameByName(self.tcp_frame_name, self.arm_model_index)

        self.base_frame: Frame = self.plant.GetFrameByName("base", self.arm_model_index)

        self.meshcat = meshcat

    def forward_kinematics(self, q: JointConfigurationType) -> HomogeneousMatrixType:
        root_context = self.robot_diagram.CreateDefaultContext()
        plant_context = self.robot_diagram.plant_context(root_context)
        self.plant.SetPositions(plant_context, self.arm_model_index, q)
        X_B_TCP = self.tcp_frame.CalcPose(plant_context, self.base_frame)

        self.robot_diagram.ForcedPublish(root_context)

        return X_B_TCP.GetAsMatrix4()

    def inverse_kinematics(
        self,
        X_RobotBase_Tcp: Union[RigidTransform, HomogeneousMatrixType],
        q_init: JointConfigurationType | None,
        override_position_tolerance: List[List[float]] | None = None,
        override_rotation_tolerance: float | None = None,
    ) -> List[JointConfigurationType] | None:
        if self.meshcat is not None:
            X_W_RobotBase = self.plant.GetFrameByName("base", self.arm_model_index).CalcPoseInWorld(
                self.plant.CreateDefaultContext()
            )
            X_W_TCP = X_W_RobotBase @ RigidTransform(X_RobotBase_Tcp)

            add_meshcat_triad(
                self.meshcat, "IK/IK_pose", length=0.05, radius=0.002, opacity=1.0, X_W_Triad=X_W_TCP, rgba_xyz=None
            )

        for q in [q_init, *self.ur_plausible_configurations]:
            if q is not None:
                q_target = self.inverse_kinematics_from_q0(
                    q, X_RobotBase_Tcp, override_position_tolerance, override_rotation_tolerance
                )
                if q_target is not None and np.all(1 - np.isnan(q_target)):
                    return [q_target]

        logger.warning("Failed to find a solution for the IK problem")
        return None

    def inverse_kinematics_from_q0(
        self,
        q0: JointConfigurationType,
        X_RobotBase_Tcp: Union[RigidTransform, HomogeneousMatrixType],
        override_position_tolerance: List[List[float]] | None = None,
        override_rotation_tolerance: float | None = None,
        ignore_collisions: bool = False,
    ) -> List[JointConfigurationType] | None:
        root_context = self.robot_diagram.CreateDefaultContext()
        context = self.plant.GetMyContextFromRoot(root_context)
        self.plant.SetPositions(context, self.arm_model_index, q0)

        # for model in self.models_to_freeze:
        #     for joint_idx in self.plant.GetJointIndices(model):
        #         self.plant.get_mutable_joint(joint_idx).Lock(context)

        # differentiate between q0, which is only arm positions, while q0_full will also contain the gripper joints
        q0_full = self.plant.GetPositions(context).copy()
        ik = inverse_kinematics.InverseKinematics(self.plant, context)
        q_variables = ik.q()
        prog = ik.prog()

        ##########################################
        # OBJECTIVE FUNCTION
        prog.AddQuadraticErrorCost(np.identity(len(q_variables)), q0_full, q_variables)
        prog.SetInitialGuess(q_variables, q0_full)

        # CONSTRAINTS
        X_P_C = RigidTransform(X_RobotBase_Tcp)
        if override_position_tolerance is None:
            position_tolerance = [[0.001, 0.001, 0.001], [0.001, 0.001, 0.001]]
        else:
            position_tolerance = override_position_tolerance
        if override_rotation_tolerance is None:
            rotation_tolerance = math.radians(3)
        else:
            rotation_tolerance = override_rotation_tolerance

        # position_tolerance = RobotKinematics.default_position_tolerance
        # rotation_tolerance = RobotKinematics.default_rotation_tolerance

        self._add_position_constraint(X_P_C, position_tolerance, ik)
        self._add_orientation_constraint(X_P_C, rotation_tolerance, ik)

        keep_a_distance_of = 0.005
        consider_objects_within_range = 0.01

        if not ignore_collisions:
            ik.AddMinimumDistanceLowerBoundConstraint(keep_a_distance_of, consider_objects_within_range)

        prog = ik.prog()

        ################################################
        # SOLVE with three solvers until solution is found or all solvers failed
        q_target = np.full(len(q_variables), np.nan)
        if SnoptSolver().available():
            solvers = [SnoptSolver(), IpoptSolver()]
        else:
            logger.warning("SNOPT not available, using IPOPT only")
            solvers = [IpoptSolver()]

        for solver in solvers:
            result = solver.Solve(prog, q0_full)
            q_target = result.GetSolution(q_variables)

            if not result.is_success():
                failed_constraints = result.GetInfeasibleConstraintNames(prog, None)
                failed_constraints = [f"\t{failedcstr}" for failedcstr in failed_constraints]
                failed_constraints_str = "\t\n".join(failed_constraints)
                logger.trace(
                    f"Solver '{result.get_solver_id().name()}' for IK problem failed constraints"
                    f"\n{failed_constraints_str}\n will potentially retry with another solver."
                )

        if not result.is_success():
            logger.warning(
                f"Failed to find a solution for the IK problem (coming from q_init: {q0}),"
                f" will give closest solution instead"
            )

        self.plant.SetPositions(context, q_target)
        q_target_arm = self.plant.GetPositions(context, self.arm_model_index).copy()
        self.robot_diagram.ForcedPublish(root_context)

        # lets not return a result when failed
        if not result.is_success():
            q_target_arm = None
        return q_target_arm

    def _add_position_constraint(self, X_RobotBase_Tcp: RigidTransform, pos_tolerance, ik):
        """Add position constraint to the ik problem. Implements an inequality
        constraint where f_p(q) must lie between p_WG_lower and p_WG_upper.
        Can be translated to ik.prog().AddBoundingBoxConstraint(f_p(q), p_WG_lower, p_WG_upper)
        """
        pos_tol_lower = pos_tolerance[0]
        pos_tol_upper = pos_tolerance[1]
        p_Parent_TCP_lower = X_RobotBase_Tcp.translation().copy() - pos_tol_lower
        p_Parent_TCP_upper = X_RobotBase_Tcp.translation().copy() + pos_tol_upper

        ik.AddPositionConstraint(
            frameA=self.base_frame,
            frameB=self.tcp_frame,
            p_BQ=np.zeros(3),
            p_AQ_lower=p_Parent_TCP_lower,
            p_AQ_upper=p_Parent_TCP_upper,
        )

    def _add_orientation_constraint(self, X_RobotBase_Tcp, rot_tolerance, ik):
        """Add orientation constraint to the ik problem. Implements an inequality
        constraint where the axis-angle difference between f_R(q) and R_WG must be
        within bounds. Can be translated to:
        ik.prog().AddBoundingBoxConstraint(angle_diff(f_R(q), R_WG), -bounds, bounds)
        """
        R_WG = X_RobotBase_Tcp.rotation()

        ik.AddOrientationConstraint(
            frameAbar=self.base_frame,
            R_AbarA=R_WG,
            frameBbar=self.tcp_frame,
            R_BbarB=RotationMatrix(),
            theta_bound=rot_tolerance,
        )
