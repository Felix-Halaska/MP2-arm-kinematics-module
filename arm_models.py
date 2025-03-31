from math import sin, cos, acos, atan2, sqrt, asin
import math
import numpy as np
from matplotlib.figure import Figure
from helper_fcns.utils import EndEffector, rotm_to_euler, near_zero, dh_to_matrix
import sympy as sp
from sympy import evalf
import helper_fcns.utils as ut

PI = 3.1415926535897932384
np.set_printoptions(precision=3)


class Robot:
    """
    Represents a robot manipulator with various kinematic configurations.
    Provides methods to calculate forward kinematics, inverse kinematics, and velocity kinematics.
    Also includes methods to visualize the robot's motion and state in 3D.

    Attributes:
        num_joints (int): Number of joints in the robot.
        ee_coordinates (list): List of end-effector coordinates.
        robot (object): The robot object (e.g., TwoDOFRobot, ScaraRobot, etc.).
        origin (list): Origin of the coordinate system.
        axes_length (float): Length of the axes for visualization.
        point_x, point_y, point_z (list): Lists to store coordinates of points for visualization.
        show_animation (bool): Whether to show the animation or not.
        plot_limits (list): Limits for the plot view.
        fig (matplotlib.figure.Figure): Matplotlib figure for 3D visualization.
        sub1 (matplotlib.axes._subplots.Axes3DSubplot): Matplotlib 3D subplot.
    """

    def __init__(self, type="2-dof", show_animation: bool = True):
        """
        Initializes a robot with a specific configuration based on the type.

        Args:
            type (str, optional): Type of robot (e.g., '2-dof', 'scara', '5-dof'). Defaults to '2-dof'.
            show_animation (bool, optional): Whether to show animation of robot movement. Defaults to True.
        """
        if type == "2-dof":
            self.num_joints = 2
            self.ee_coordinates = ["X", "Y"]
            self.robot = TwoDOFRobot()

        elif type == "scara":
            self.num_joints = 3
            self.ee_coordinates = ["X", "Y", "Z", "Theta"]
            self.robot = ScaraRobot()

        elif type == "5-dof":
            self.num_joints = 5
            self.ee_coordinates = ["X", "Y", "Z", "RotX", "RotY", "RotZ"]
            self.robot = FiveDOFRobot()

        self.origin = [0.0, 0.0, 0.0]
        self.axes_length = 0.075
        self.point_x, self.point_y, self.point_z = [], [], []
        self.show_animation = show_animation
        self.plot_limits = [0.75, 0.75, 1.0]

        if self.show_animation:
            self.fig = Figure(figsize=(12, 10), dpi=100)
            self.sub1 = self.fig.add_subplot(1, 1, 1, projection="3d")
            self.fig.suptitle("Manipulator Kinematics Visualization", fontsize=16)

        # initialize figure plot
        self.init_plot()

    def init_plot(self):
        """Initializes the plot by calculating the robot's points and calling the plot function."""
        self.robot.calc_robot_points()
        self.plot_3D()

    def update_plot(self, pose=None, angles=None, soln=0, numerical=False):
        """
        Updates the robot's state based on new pose or joint angles and updates the visualization.

        Args:
            pose (EndEffector, optional): Desired end-effector pose for inverse kinematics.
            angles (list, optional): Joint angles for forward kinematics.
            soln (int, optional): The inverse kinematics solution to use (0 or 1).
            numerical (bool, optional): Whether to use numerical inverse kinematics.
        """
        if pose is not None:  # Inverse kinematics case
            if not numerical:
                self.robot.calc_inverse_kinematics(pose, soln=soln)
            else:
                self.robot.calc_numerical_ik(pose, tol=0.02, ilimit=100)
        elif angles is not None:  # Forward kinematics case
            self.robot.calc_forward_kinematics(angles, radians=False)
        else:
            return
        self.plot_3D()

    def move_velocity(self, vel):
        """
        Moves the robot based on a given velocity input.

        Args:
            vel (list): Velocity input for the robot.
        """
        self.robot.calc_velocity_kinematics(vel)
        self.plot_3D()

    def draw_line_3D(self, p1, p2, format_type: str = "k-"):
        """
        Draws a 3D line between two points.

        Args:
            p1 (list): Coordinates of the first point.
            p2 (list): Coordinates of the second point.
            format_type (str, optional): The format of the line. Defaults to "k-".
        """
        self.sub1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], format_type)

    def draw_ref_line(self, point, axes=None, ref="xyz"):
        """
        Draws reference lines from a given point along specified axes.

        Args:
            point (list): The coordinates of the point to draw from.
            axes (matplotlib.axes, optional): The axes on which to draw the reference lines.
            ref (str, optional): Which reference axes to draw ('xyz', 'xy', or 'xz'). Defaults to 'xyz'.
        """
        line_width = 0.7
        if ref == "xyz":
            axes.plot(
                [point[0], self.plot_limits[0]],
                [point[1], point[1]],
                [point[2], point[2]],
                "b--",
                linewidth=line_width,
            )  # X line
            axes.plot(
                [point[0], point[0]],
                [point[1], self.plot_limits[1]],
                [point[2], point[2]],
                "b--",
                linewidth=line_width,
            )  # Y line
            axes.plot(
                [point[0], point[0]],
                [point[1], point[1]],
                [point[2], 0.0],
                "b--",
                linewidth=line_width,
            )  # Z line
        elif ref == "xy":
            axes.plot(
                [point[0], self.plot_limits[0]],
                [point[1], point[1]],
                "b--",
                linewidth=line_width,
            )  # X line
            axes.plot(
                [point[0], point[0]],
                [point[1], self.plot_limits[1]],
                "b--",
                linewidth=line_width,
            )  # Y line
        elif ref == "xz":
            axes.plot(
                [point[0], self.plot_limits[0]],
                [point[2], point[2]],
                "b--",
                linewidth=line_width,
            )  # X line
            axes.plot(
                [point[0], point[0]], [point[2], 0.0], "b--", linewidth=line_width
            )  # Z line

    def plot_3D(self):
        """
        Plots the 3D visualization of the robot, including the robot's links, end-effector, and reference frames.
        """
        self.sub1.cla()
        self.point_x.clear()
        self.point_y.clear()
        self.point_z.clear()

        EE = self.robot.ee

        # draw lines to connect the points
        for i in range(len(self.robot.points) - 1):
            self.draw_line_3D(self.robot.points[i], self.robot.points[i + 1])

        # draw the points
        for i in range(len(self.robot.points)):
            self.point_x.append(self.robot.points[i][0])
            self.point_y.append(self.robot.points[i][1])
            self.point_z.append(self.robot.points[i][2])
        self.sub1.plot(
            self.point_x,
            self.point_y,
            self.point_z,
            marker="o",
            markerfacecolor="m",
            markersize=12,
        )

        # draw the EE
        self.sub1.plot(EE.x, EE.y, EE.z, "bo")
        # draw the base reference frame
        self.draw_line_3D(
            self.origin,
            [self.origin[0] + self.axes_length, self.origin[1], self.origin[2]],
            format_type="r-",
        )
        self.draw_line_3D(
            self.origin,
            [self.origin[0], self.origin[1] + self.axes_length, self.origin[2]],
            format_type="g-",
        )
        self.draw_line_3D(
            self.origin,
            [self.origin[0], self.origin[1], self.origin[2] + self.axes_length],
            format_type="b-",
        )
        # draw the EE reference frame
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[0], format_type="r-")
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[1], format_type="g-")
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[2], format_type="b-")
        # draw reference / trace lines
        self.draw_ref_line([EE.x, EE.y, EE.z], self.sub1, ref="xyz")

        # add text at bottom of window
        pose_text = "End-effector Pose:      [ "
        pose_text += f"X: {round(EE.x,4)},  "
        pose_text += f"Y: {round(EE.y,4)},  "
        pose_text += f"Z: {round(EE.z,4)},  "
        pose_text += f"RotX: {round(EE.rotx,4)},  "
        pose_text += f"RotY: {round(EE.roty,4)},  "
        pose_text += f"RotZ: {round(EE.rotz,4)}  "
        pose_text += " ]"

        theta_text = "Joint Positions (deg/m):     ["
        for i in range(self.num_joints):
            theta_text += f" {round(np.rad2deg(self.robot.theta[i]),2)}, "
        theta_text += " ]"

        textstr = pose_text + "\n" + theta_text
        self.sub1.text2D(
            0.2, 0.02, textstr, fontsize=13, transform=self.fig.transFigure
        )

        self.sub1.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        self.sub1.set_ylim(-self.plot_limits[1], self.plot_limits[1])
        self.sub1.set_zlim(0, self.plot_limits[2])
        self.sub1.set_xlabel("x [m]")
        self.sub1.set_ylabel("y [m]")


class TwoDOFRobot:
    """
    Represents a 2-degree-of-freedom (DOF) robot arm with two joints and one end effector.
    Includes methods for calculating forward kinematics (FPK), inverse kinematics (IPK),
    and velocity kinematics (VK).

    Attributes:
        l1 (float): Length of the first arm segment.
        l2 (float): Length of the second arm segment.
        theta (list): Joint angles.
        theta_limits (list): Joint limits for each joint.
        ee (EndEffector): The end effector object.
        points (list): List of points representing the robot's configuration.
        num_dof (int): The number of degrees of freedom (2 for this robot).
    """

    def __init__(self):
        """
        Initializes a 2-DOF robot with default arm segment lengths and joint angles.
        """
        self.l1 = 0.30  # Length of the first arm segment
        self.l2 = 0.25  # Length of the second arm segment

        self.theta = [0.0, 0.0]  # Joint angles (in radians)
        self.theta_limits = [[-PI, PI], [-PI + 0.261, PI - 0.261]]  # Joint limits

        self.ee = EndEffector()  # The end-effector object
        self.num_dof = 2  # Number of degrees of freedom
        self.points = [None] * (self.num_dof + 1)  # List to store robot points

    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculates the forward kinematics for the robot based on the joint angles.

        Args:
            theta (list): Joint angles.
            radians (bool, optional): Whether the angles are in radians or degrees. Defaults to False.
        """

        if not radians:
            # Convert degrees to radians if the input is in degrees
            self.theta[0] = np.deg2rad(theta[0])
            self.theta[1] = np.deg2rad(theta[1])
        else:
            self.theta = theta

        # Ensure that the joint angles respect the joint limits
        for i, th in enumerate(self.theta):
            self.theta[i] = np.clip(
                th, self.theta_limits[i][0], self.theta_limits[i][1]
            )

        # Update the robot configuration (i.e., the positions of the joints and end effector)
        self.calc_robot_points()

    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculates the inverse kinematics (IK) for a given end effector position.

        Args:
            EE (EndEffector): The end effector object containing the target position (x, y).
            soln (int, optional): The solution branch to use. Defaults to 0 (first solution).
        """
        x, y = EE.x, EE.y
        l1, l2 = self.l1, self.l2

        L = sqrt(x**2 + y**2)
        beta = acos((l1**2 + l2**2 - L**2) / (2 * l1 * l2))
        alpha = np.arctan2(y, x)
        phi = asin((l2 * sin(np.pi - beta)) / L)

        if soln == 0:
            self.theta[0] = alpha + phi
            self.theta[1] = np.pi + beta
        elif soln == 1:
            self.theta[0] = alpha - phi
            self.theta[1] = np.pi - beta

        # Calculate robot points based on the updated joint angles
        self.calc_robot_points()

    def calc_numerical_ik(self, EE: EndEffector, tol=0.01, ilimit=50):
        """
        Calculates numerical inverse kinematics (IK) based on input end effector coordinates.

        Args:
            EE (EndEffector): The end effector object containing the target position (x, y).
            tol (float, optional): The tolerance for the solution. Defaults to 0.01.
            ilimit (int, optional): The maximum number of iterations. Defaults to 50.
        """

        # x, y are desired, self.ee.x and self.ee.y are current
        x, y = EE.x, EE.y

        error = [x - self.ee.x, y - self.ee.y]

        print(x, y)
        print(self.ee.x, self.ee.y)
        print(f"error: {error}")
        new_theta = self.theta
        print(new_theta)
        for i in range(ilimit):
            if abs(sqrt(error[0] ** 2 + error[1] ** 2)) > tol:
                error = [x - self.ee.x, y - self.ee.y]
                new_theta = new_theta + self.inverse_jacobian() @ error
                self.theta = new_theta
                print(f"theta: {self.theta}")
                self.calc_robot_points()
            else:
                self.theta = new_theta
        self.calc_robot_points()

    def calc_velocity_kinematics(self, vel: list):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy].
        """

        ########################################

        # insert your code here

        ########################################

        # Update robot points based on the new joint angles
        self.calc_robot_points()

    def jacobian(self, theta: list = None):
        """
        Returns the Jacobian matrix for the robot. If theta is not provided,
        the function will use the object's internal theta attribute.

        Args:
            theta (list, optional): The joint angles for the robot. Defaults to self.theta.

        Returns:
            np.ndarray: The Jacobian matrix (2x2).
        """
        # Use default values if arguments are not provided
        if theta is None:
            theta = self.theta

        return np.array(
            [
                [
                    -self.l1 * sin(theta[0]) - self.l2 * sin(theta[0] + theta[1]),
                    -self.l2 * sin(theta[0] + theta[1]),
                ],
                [
                    self.l1 * cos(theta[0]) + self.l2 * cos(theta[0] + theta[1]),
                    self.l2 * cos(theta[0] + theta[1]),
                ],
            ]
        )

    def inverse_jacobian(self):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        J = self.jacobian()
        print(f"Determinant of J: {np.linalg.det(J)}")
        # return np.linalg.inv(self.jacobian())
        return np.linalg.pinv(self.jacobian())

    def calc_robot_points(self):
        """
        Calculates the positions of the robot's joints and the end effector.

        Updates the `points` list, storing the coordinates of the base, shoulder, elbow, and end effector.
        """

        ########################################

        # Replace the placeholder values with your code

        placeholder_value = [0.0, 0.0, 0.0]

        # Base position
        self.points[0] = [0.0, 0.0, 0.0]
        # Shoulder joint
        self.points[1] = [
            self.l1 * cos(self.theta[0]),
            self.l1 * sin(self.theta[0]),
            0.0,
        ]
        # Elbow joint
        self.points[2] = [
            self.l1 * cos(self.theta[0]) + self.l2 * cos(self.theta[0] + self.theta[1]),
            self.l1 * sin(self.theta[0]) + self.l2 * sin(self.theta[0] + self.theta[1]),
            0.0,
        ]

        ########################################

        # Update end effector position
        self.ee.x = self.points[2][0]
        self.ee.y = self.points[2][1]
        self.ee.z = self.points[2][2]
        self.ee.rotz = self.theta[0] + self.theta[1]

        # End effector axes
        self.EE_axes = np.zeros((3, 3))
        self.EE_axes[0] = (
            np.array(
                [
                    cos(self.theta[0] + self.theta[1]),
                    sin(self.theta[0] + self.theta[1]),
                    0,
                ]
            )
            * 0.075
            + self.points[2]
        )
        self.EE_axes[1] = (
            np.array(
                [
                    -sin(self.theta[0] + self.theta[1]),
                    cos(self.theta[0] + self.theta[1]),
                    0,
                ]
            )
            * 0.075
            + self.points[2]
        )
        self.EE_axes[2] = np.array([0, 0, 1]) * 0.075 + self.points[2]


class ScaraRobot:
    """
    A class representing a SCARA (Selective Compliance Assembly Robot Arm) robot.
    This class handles the kinematics (forward, inverse, and velocity kinematics)
    and robot configuration, including joint limits and end-effector calculations.
    """

    def __init__(self):
        """
        Initializes the SCARA robot with its geometry, joint variables, and limits.
        Sets up the transformation matrices and robot points.
        """
        # Geometry of the robot (link lengths in meters)
        self.l1 = 0.35  # Base to 1st joint
        self.l2 = 0.18  # 1st joint to 2nd joint
        self.l3 = 0.15  # 2nd joint to 3rd joint
        self.l4 = 0.30  # 3rd joint to 4th joint (tool or end-effector)
        self.l5 = 0.12  # Tool offset

        # Joint variables (angles in radians)
        self.theta = [0.0, 0.0, 0.0]

        # Joint angle limits (min, max) for each joint
        self.theta_limits = [
            [-np.pi, np.pi],
            [-np.pi + 0.261, np.pi - 0.261],
            [0, self.l1 + self.l3 - self.l5],
        ]

        # End-effector (EE) object to store EE position and orientation
        self.ee = EndEffector()

        # Number of degrees of freedom and number of points to store robot configuration
        self.num_dof = 3
        self.num_points = 7
        self.points = [None] * self.num_points

        # Transformation matrices (DH parameters and resulting transformation)
        self.DH = np.zeros((5, 4))  # Denavit-Hartenberg parameters (theta, d, a, alpha)
        self.T = np.zeros((self.num_dof, 4, 4))  # Transformation matrices

        ########################################

        # insert your additional code here

        ########################################

    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculate Forward Kinematics (FK) based on the given joint angles.

        Args:
            theta (list): Joint angles (in radians if radians=True, otherwise in degrees).
            radians (bool): Whether the input angles are in radians (default is False).
        """
        ########################################

        # insert your code here

        ########################################

        # Calculate robot points (e.g., end-effector position)
        self.calc_robot_points()

    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculate Inverse Kinematics (IK) based on the input end-effector coordinates.

        Args:
            EE (EndEffector): End-effector object containing desired position (x, y, z).
            soln (int): Solution index (0 or 1), for multiple possible IK solutions.
        """
        x, y, z = EE.x, EE.y, EE.z
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5

        L = sqrt(x**2 + y**2)
        beta = acos((l2**2 + l4**2 - L**2) / (2 * l2 * l4))
        alpha = np.arctan2(y, x)
        phi = asin((l4 * sin(np.pi - beta)) / L)

        if soln == 0:
            self.theta[0] = alpha + phi
            self.theta[1] = np.pi + beta
        elif soln == 1:
            self.theta[0] = alpha - phi
            self.theta[1] = np.pi - beta

        self.theta[2] = l1 + l3 - l5 - z
        ####################################

        # Recalculate Forward Kinematics to update the robot's configuration
        self.calc_forward_kinematics(self.theta, radians=True)

    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate velocity kinematics and update joint velocities.

        Args:
            vel (array): Linear velocities (3D) of the end-effector.
        """
        ########################################

        # insert your code here

        ########################################

        # Recalculate robot points based on updated joint angles
        self.calc_robot_points()

    def calc_robot_points(self):
        """
        Calculate the main robot points (links and end-effector position) using the current joint angles.
        Updates the robot's points array and end-effector position.
        """

        # Calculate transformation matrices for each joint and end-effector
        self.points[0] = np.array([0, 0, 0, 1])
        self.points[1] = np.array([0, 0, self.l1, 1])
        self.points[2] = self.T[0] @ self.points[0]
        self.points[3] = self.points[2] + np.array([0, 0, self.l3, 1])
        self.points[4] = self.T[0] @ self.T[1] @ self.points[0] + np.array(
            [0, 0, self.l5, 1]
        )
        self.points[5] = self.T[0] @ self.T[1] @ self.points[0]
        self.points[6] = self.T[0] @ self.T[1] @ self.T[2] @ self.points[0]

        self.EE_axes = (
            self.T[0] @ self.T[1] @ self.T[2] @ np.array([0.075, 0.075, 0.075, 1])
        )
        self.T_ee = self.T[0] @ self.T[1] @ self.T[2]

        # End-effector (EE) position and axes
        self.ee.x = self.points[-1][0]
        self.ee.y = self.points[-1][1]
        self.ee.z = self.points[-1][2]
        rpy = rotm_to_euler(self.T_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy

        # EE coordinate axes
        self.EE_axes = np.zeros((3, 3))
        self.EE_axes[0] = self.T_ee[:3, 0] * 0.075 + self.points[-1][0:3]
        self.EE_axes[1] = self.T_ee[:3, 1] * 0.075 + self.points[-1][0:3]
        self.EE_axes[2] = self.T_ee[:3, 2] * 0.075 + self.points[-1][0:3]


class FiveDOFRobot:
    """
    A class to represent a 5-DOF robotic arm with kinematics calculations, including
    forward kinematics, inverse kinematics, velocity kinematics, and Jacobian computation.

    Attributes:
        l1, l2, l3, l4, l5: Link lengths of the robotic arm.
        theta: List of joint angles in radians.
        theta_limits: Joint limits for each joint.
        ee: End-effector object for storing the position and orientation of the end-effector.
        num_dof: Number of degrees of freedom (5 in this case).
        points: List storing the positions of the robot joints.
        DH: Denavit-Hartenberg parameters for each joint.
        T: Transformation matrices for each joint.
    """

    def __init__(self):
        """Initialize the robot parameters and joint limits."""
        # Link lengths
        self.l1, self.l2, self.l3, self.l4, self.l5 = 0.30, 0.15, 0.18, 0.15, 0.12

        # Joint angles (initialized to zero)
        self.theta = [0, 0, 0, 0, 0]

        # Joint limits (in radians)
        self.theta_limits = [
            [-np.pi, np.pi],
            [-np.pi / 3, np.pi],
            [-np.pi + np.pi / 12, np.pi - np.pi / 4],
            [-np.pi + np.pi / 12, np.pi - np.pi / 12],
            [-np.pi, np.pi],
        ]

        # End-effector object
        self.ee = EndEffector()

        # Robot's points
        self.num_dof = 5
        self.points = [None] * (self.num_dof + 1)

        # Denavit-Hartenberg parameters and transformation matrices
        self.DH = np.zeros((5, 4))
        self.T = np.zeros((self.num_dof, 4, 4))

        ########################################

        # insert your additional code here

        ########################################
        self.calc_HTM()

    def calc_HTM(self):

        arr = []
        dhTable = [
            [self.theta[0], self.l1, 0, math.pi / 2],
            [math.pi / 2, 0, 0, 0],
            [self.theta[1], 0, self.l2, math.pi],
            [self.theta[2], 0, self.l3, math.pi],
            [self.theta[3], 0, self.l4, 0],
            [-math.pi / 2, 0, 0, -math.pi / 2],
            [self.theta[4], self.l5, 0, 0],
        ]

        for i in range(len(dhTable)):
            arr.append(ut.dh_to_matrix(dhTable[i]))

        self.T[0] = np.matmul(arr[0], arr[1])
        self.T[1] = arr[2]
        self.T[2] = arr[3]
        self.T[3] = np.matmul(arr[4], arr[5])
        self.T[4] = arr[6]  # m45 is just the rotation so it does not need another link
        # print (self.T)

    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculate forward kinematics based on the provided joint angles.

        Args:
            theta: List of joint angles (in degrees or radians).
            radians: Boolean flag to indicate if input angles are in radians.
        """
        if not radians:
            for i in range(len(self.theta)):
                self.theta[i] = theta[i] * PI / 180
        else:
            self.theta = theta

        # Update the robot configuration (i.e., the positions of the joints and end effector)
        self.calc_robot_points()

    def calc_inverse_kinematics(self, EE: EndEffector, soln=0, radians=False):
        """
        Calculate inverse kinematics to determine the joint angles based on end-effector position.

        Args:
            EE: EndEffector object containing desired position and orientation.
            soln: Optional parameter for multiple solutions (not implemented).
        """
        # Redefine EE positions
        x, y, z = EE.x, EE.y, EE.z

        # Find wrist position
        R_EE = np.array(ut.euler_to_rotm((EE.rotx, EE.roty, EE.rotz)), dtype=float)
        z_zero = np.transpose(np.array([[0, 0, 1]]))
        p_wrist = np.array([[x], [y], [z]]) - ((self.l4 + self.l5) * (R_EE @ z_zero))

        x_wrist = p_wrist[0]
        y_wrist = p_wrist[1]
        z_wrist = p_wrist[2]

        # Calculate theta 1
        t_1 = atan2(y_wrist, x_wrist) + math.pi

        # Define parallel plane
        s = z_wrist - self.l1
        r = sqrt(x_wrist**2 + y_wrist**2)

        # Find geometric relationships
        L = sqrt(s**2 + r**2)
        alpha = atan2(r, s)
        beta = np.arccos(
            np.clip((self.l2**2 + self.l3**2 - L**2) / (2 * self.l2 * self.l3), -1, 1)
        )

        # Solve for two solutions for theta 2 and theta 3
        if soln == 0:
            t_3 = math.pi - beta
            phi = atan2((self.l3 * sin(-t_3)), (self.l2 + (self.l3 * cos(-t_3))))
            t_2 = alpha - phi
        elif soln == 1:
            t_3 = -math.pi + beta
            phi = atan2((self.l3 * sin(-t_3)), (self.l2 + (self.l3 * cos(-t_3))))
            t_2 = alpha - phi

        R_1 = np.array([[cos(t_1),0,sin(t_1)],[sin(t_1),0,-cos(t_1)],[0,1,0]])
        R_1_5 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        R_2 = np.array([[cos(t_2),sin(t_2),0],[sin(t_2),-cos(t_2),0],[0,0,-1]])
        R_3 = np.array([[cos(t_3),sin(t_3),0],[sin(t_3),-cos(t_3),0],[0,0,-1]])
        R_0_3 = R_1 @ R_1_5 @ R_2 @ R_3

        R_3_5 = np.transpose(R_0_3) @ R_EE

        t_4 = atan2(R_3_5[1,2],R_3_5[0,2])
        t_5 = atan2(-R_3_5[2,0],-R_3_5[2,1])

        #Assign theta values to theta list
        # Assign theta values to theta list
        self.theta[0] = t_1
        self.theta[1] = t_2
        self.theta[2] = t_3
        self.theta[3] = t_4
        self.theta[4] = t_5


        # Solve for alternative solutions
        t_1_1 = atan2(y_wrist, x_wrist)
        r_1 = -1 * sqrt(x_wrist**2 + y_wrist**2)
        alpha_1 = atan2(r_1, s)
        t_3 = math.pi - beta
        t_3_1 = -math.pi + beta

        phi = atan2((self.l3 * sin(-t_3)), (self.l2 + (self.l3 * cos(-t_3))))
        phi_1 = atan2((self.l3 * sin(-t_3_1)), (self.l2 + (self.l3 * cos(-t_3_1))))

        #OK SOLUTIONS HERE WE GO

        # s = z_wrist - self.l1
        # i = 0

        # theta_1_list = [0, math.pi]
        # r_list = [1, -1]
        # theta_3_list = [1, -1]

        # solutions = np.zeros((8,6))

        # for theta_1_option in theta_1_list:
        #     theta_1 = atan2(y_wrist, x_wrist) + theta_1_option

        #     for r_option in r_list:
        #         r = r_option * sqrt(x_wrist**2 + y_wrist**2)
        #         L = sqrt(s**2 + r**2)
        #         alpha = atan2(r,s) 
        #         beta = np.arccos(np.clip((self.l2**2 + self.l3**2 - L**2)/(2*self.l2*self.l3),-1,1))

        #         for theta_3_option in theta_3_list:
        #             theta_3 = theta_3_option * (math.pi - beta)
        #             phi = atan2((self.l3*sin(-theta_3)),(self.l2+(self.l3*cos(-theta_3))))
        #             theta_2 = alpha - phi

        #             R_0_1 = np.array([[cos(theta_1),0,sin(theta_1)],[sin(theta_1),0,-cos(theta_1)],[0,1,0]])
        #             R_1_15 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        #             R_15_2 = np.array([[cos(theta_2),sin(theta_2),0],[sin(theta_2),-cos(theta_2),0],[0,0,-1]])
        #             R_2_3 = np.array([[cos(theta_3),sin(theta_3),0],[sin(theta_3),-cos(theta_3),0],[0,0,-1]])
        #             R_03 = R_0_1 @ R_1_15 @ R_15_2 @ R_2_3

        #             R_35 = np.transpose(R_03) @ R_EE

        #             theta_4 = atan2(R_35[1,2],R_35[0,2])
        #             theta_5 = atan2(-R_35[2,0],-R_35[2,1])

        #             solutions[i,0:6] = [theta_1, theta_2, theta_3, theta_4, theta_5]

        #             self.calc_HTM()
        #             EE_orientation = ut.rotm_to_euler(self.T[0:4,0:4])
                        ####HOW TO CALCULATE TARGET AND THEORETICAL EE POS AND ORIENTATION
                    
        #             if solutions[i,0] >= self.theta_limits[0,0] and solutions[i,0] <= self.theta_limits[0,1] and 
        #                 solutions[i,1] >= self.theta_limits[1,0] and solutions[i,1] <= self.theta_limits[1,1] and
        #                 solutions[i,2] >= self.theta_limits[2,0] and solutions[i,2] <= self.theta_limits[2,1] and
        #                 solutions[i,3] >= self.theta_limits[3,0] and solutions[i,3] <= self.theta_limits[3,1] and
        #                 solutions[i,4] >= self.theta_limits[4,0] and solutions[i,4] <= self.theta_limits[4,1] and


        #             i = i+1


        # Assign all possible solutions to a list
        # solutions = np.zeros(8, 5)
        # solutions[0, 0:2] = [t_1_1, t_2, t_3]
        # solutions[1, 0] = [t_1_1, t_2_2, t_3]
        # solutions[2, 0] = [t_1_1, t_2_1, t_3_1]
        # solutions[3, 0] = [t_1_1, t_2_3, t_3_1]
        # solutions[4, 0] = [t_1, t_2, t_3]
        # solutions[5, 0] = [t_1, t_2_2, t_3]
        # solutions[6, 0] = [t_1, t_2_1, t_3_1]
        # solutions[7, 0] = [t_1, t_2_3, t_3, 1]

        self.calc_robot_points()

        # L = sqrt(x**2 + y**2)
        # beta = acos((l1**2 + l2**2 - L**2)/(2*l1*l2))
        # alpha = np.arctan2(y,x)
        # phi = asin((l2*sin(np.pi-beta))/L)

        # if soln == 0:
        #     self.theta[0] = alpha + phi
        #     self.theta[1] = np.pi + beta
        # elif soln == 1:
        #     self.theta[0] = alpha - phi
        #     self.theta[1] = np.pi - beta

    def calc_numerical_ik(self, EE: EndEffector, tol=0.01, ilimit=100):
        """
        Calculates numerical inverse kinematics (IK) based on input end effector coordinates.

        Args:
            EE (EndEffector): The end effector object containing the target position (x, y).
            tol (float, optional): The tolerance for the solution. Defaults to 0.01.
            ilimit (int, optional): The maximum number of iterations. Defaults to 50.
        """

        # x, y, z are desired, self.ee.x, self.ee.y, and self.ee.z are current
        x, y, z = EE.x, EE.y, EE.z

        error = [x - self.ee.x, y - self.ee.y, z - self.ee.z]
        for i in range(ilimit):
            if sqrt(error[0] ** 2 + error[1] ** 2 + error[2] ** 2) > tol:
                error = [x - self.ee.x, y - self.ee.y, z - self.ee.z]
                self.theta = self.theta + self.inverse_jacobian() @ error
                count = 0
                for angle in self.theta:
                    if angle < self.theta_limits[count][0]:
                        self.theta[count] = self.theta_limits[count][0]
                    if angle > self.theta_limits[count][1]:
                        self.theta[count] = self.theta_limits[count][1]
                    count = count + 1
                self.calc_robot_points()
                print(error)
            else:
                break
        self.calc_robot_points()

    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate the joint velocities required to achieve the given end-effector velocity.

        Args:
            vel: Desired end-effector velocity (3x1 vector).
        """
        t0, t1, t2, t3, t4 = sp.symbols("t0 t1 t2 t3 t4 ")

        dhTable = [
            [t0, self.l1, 0, math.pi / 2],
            [math.pi / 2, 0, 0, 0],
            [t1, 0, self.l2, math.pi],
            [t2, 0, self.l3, math.pi],
            [t3, 0, self.l4, 0],
            [-math.pi / 2, 0, 0, -math.pi / 2],
            [t4, self.l5, 0, 0],
        ]

        # create our homogeneous tranformation matrix symbolically
        arr = []
        for i in range(len(dhTable)):
            arr.append(ut.dh_sympi_to_matrix(dhTable[i]))

        Hm = arr[0] * arr[1] * arr[2] * arr[3] * arr[4] * arr[5] * arr[6]

        Hx = Hm[0, 3]
        Hy = Hm[1, 3]
        Hz = Hm[2, 3]

        # create our Jv matrix by taking the symbolic partial derivatives
        jacobian = sp.Matrix(
            [
                [
                    sp.diff(Hx, t0),
                    sp.diff(Hx, t1),
                    sp.diff(Hx, t2),
                    sp.diff(Hx, t3),
                    sp.diff(Hx, t4),
                ],
                [
                    sp.diff(Hy, t0),
                    sp.diff(Hy, t1),
                    sp.diff(Hy, t2),
                    sp.diff(Hy, t3),
                    sp.diff(Hy, t4),
                ],
                [
                    sp.diff(Hz, t0),
                    sp.diff(Hz, t1),
                    sp.diff(Hz, t2),
                    sp.diff(Hz, t3),
                    sp.diff(Hz, t4),
                ],
            ]
        )

        # substitue in real angle values
        jacobian = jacobian.evalf(subs={t0: self.theta[0]})
        jacobian = jacobian.evalf(subs={t1: self.theta[1]})
        jacobian = jacobian.evalf(subs={t2: self.theta[2]})
        jacobian = jacobian.evalf(subs={t3: self.theta[3]})
        jacobian = jacobian.evalf(subs={t4: self.theta[4]})

        # take the psudoinverse and add a small constant to avoid singularities
        invJac = np.array(
            sp.transpose(jacobian)
            * (((jacobian * sp.transpose(jacobian)) + sp.eye(3) * 0.0001) ** -1)
        )

        # jacobian_np = np.array(jacobian, dtype=float)
        # jac_transpose_np = np.array(sp.transpose(jacobian), dtype=float)

        # jjt = jacobian_np @ jac_transpose_np
        # det = np.linalg.det(jjt)

        vel = np.array([vel])

        thetaDot = np.matmul(invJac, np.transpose(vel))

        # calculate new angular positions
        timeStep = 0.02
        for i in range(len(self.theta)):
            self.theta[i] = self.theta[i] + (float(thetaDot[i][0]) * timeStep)

        self.calc_forward_kinematics(self.theta, radians=True)

    def jacobian(self, theta: list = None):
        """
        Compute the Jacobian matrix for the current robot configuration.

        Args:
            theta (list, optional): The joint angles for the robot. Defaults to self.theta.

        Returns:
            Jacobian matrix (3x5).
        """
        # Use default values if arguments are not provided
        if theta is None:
            theta = self.theta

        # Define DH parameters
        DH = np.zeros((5, 4))
        DH[0] = [theta[0], self.l1, 0, np.pi / 2]
        DH[1] = [theta[1] + np.pi / 2, 0, self.l2, np.pi]
        DH[2] = [theta[2], 0, self.l3, np.pi]
        DH[3] = [theta[3] - np.pi / 2, 0, 0, -np.pi / 2]
        DH[4] = [theta[4], self.l4 + self.l5, 0, 0]

        # Compute transformation matrices
        T = np.zeros((self.num_dof, 4, 4))
        for i in range(self.num_dof):
            T[i] = dh_to_matrix(DH[i])

        # Precompute transformation matrices for efficiency
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ T[i])

        # Define O0 for calculations
        O0 = np.array([0, 0, 0, 1])

        # Initialize the Jacobian matrix
        jacobian = np.zeros((3, self.num_dof))

        # Calculate the Jacobian columns
        for i in range(self.num_dof):
            T_curr = T_cumulative[i]
            T_final = T_cumulative[-1]

            # Calculate position vector r
            r = (T_final @ O0 - T_curr @ O0)[:3]

            # Compute the rotation axis z
            z = T_curr[:3, :3] @ np.array([0, 0, 1])

            # Compute linear velocity part of the Jacobian
            jacobian[:, i] = np.cross(z, r)

        # Replace near-zero values with zero, primarily for debugging purposes
        # return near_zero(jacobian)
        return jacobian

    def inverse_jacobian(self):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        J = self.jacobian()
        # print(f"Determinant of J: {np.linalg.det(J)}")
        # return np.linalg.inv(self.jacobian())
        return np.linalg.pinv(self.jacobian())

    def dh_to_matrix(self, dh_params: list) -> np.ndarray:
        """Converts Denavit-Hartenberg parameters to a transformation matrix.

        Args:
            dh_params (list): Denavit-Hartenberg parameters [theta, d, a, alpha].

        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        theta, d, a, alpha = dh_params
        return np.array(
            [
                [
                    cos(theta),
                    -sin(theta) * cos(alpha),
                    sin(theta) * sin(alpha),
                    a * cos(theta),
                ],
                [
                    sin(theta),
                    cos(theta) * cos(alpha),
                    -cos(theta) * sin(alpha),
                    a * sin(theta),
                ],
                [0, sin(alpha), cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )

    def calc_robot_points(self):
        """Calculates the main arm points using the current joint angles"""
        self.calc_HTM()
        # Initialize points[0] to the base (origin)
        self.points[0] = np.array([0, 0, 0, 1])

        # Precompute cumulative transformations to avoid redundant calculations
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Calculate the robot points by applying the cumulative transformations
        for i in range(1, 6):
            self.points[i] = T_cumulative[i] @ self.points[0]
        print(self.points[3])

        # Calculate EE position and rotation
        self.EE_axes = T_cumulative[-1] @ np.array(
            [0.075, 0.075, 0.075, 1]
        )  # End-effector axes
        self.T_ee = T_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        self.ee.x, self.ee.y, self.ee.z = self.points[-1][:3]

        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = rotm_to_euler(self.T_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy[0], rpy[1], rpy[2]

        # Calculate the EE axes in space (in the base frame)
        self.EE = [self.ee.x, self.ee.y, self.ee.z]
        self.EE_axes = np.array(
            [self.T_ee[:3, i] * 0.075 + self.points[-1][:3] for i in range(3)]
        )
