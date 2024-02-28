# env import
import gym
from gym import spaces
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
from typing import Tuple, Sequence, Dict, Union, Optional
import pygame
import pymunk
import numpy as np
import shapely.geometry as sg
import cv2
import skimage.transform as st
import pymunk.pygame_util
import collections
from matplotlib import cm
from dataset import PushTSequenceDataset
import torch
# @markdown ### **Environment**
# @markdown Defines a PyMunk-based Push-T environment `PushTEnv`.
# @markdown
# @markdown **Goal**: push the gray T-block into the green area.
# @markdown
# @markdown Adapted from [Implicit Behavior Cloning](https://implicitbc.github.io/)


positive_y_is_up: bool = False
"""Make increasing values of y point upwards.

When True::

    y
    ^
    |      . (3, 3)
    |
    |   . (2, 2)
    |
    +------ > x

When False::

    +------ > x
    |
    |   . (2, 2)
    |
    |      . (3, 3)
    v
    y

"""


def farthest_point_sampling(points: np.ndarray, n_points: int, init_idx: int):
    """
    Naive O(N^2)
    """
    assert n_points >= 1
    chosen_points = [points[init_idx]]
    for _ in range(n_points - 1):
        cpoints = np.array(chosen_points)
        all_dists = np.linalg.norm(points[:, None, :] - cpoints[None, :, :], axis=-1)
        min_dists = all_dists.min(axis=1)
        next_idx = np.argmax(min_dists)
        next_pt = points[next_idx]
        chosen_points.append(next_pt)
    result = np.array(chosen_points)
    return result


class PymunkKeypointManager:
    def __init__(
        self,
        local_keypoint_map: Dict[str, np.ndarray],
        color_map: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        local_keypoint_map:
            "<attribute_name>": (N,2) floats in object local coordinate
        """
        if color_map is None:
            cmap = cm.get_cmap("tab10")
            color_map = dict()
            for i, key in enumerate(local_keypoint_map.keys()):
                color_map[key] = (np.array(cmap.colors[i]) * 255).astype(np.uint8)

        self.local_keypoint_map = local_keypoint_map
        self.color_map = color_map

    @property
    def kwargs(self):
        return {
            "local_keypoint_map": self.local_keypoint_map,
            "color_map": self.color_map,
        }

    @classmethod
    def create_from_pusht_env(cls, env, n_block_kps=9, n_agent_kps=3, seed=0, **kwargs):
        rng = np.random.default_rng(seed=seed)
        local_keypoint_map = dict()
        for name in ["block", "agent"]:
            self = env
            self.space = pymunk.Space()
            if name == "agent":
                self.agent = obj = self.add_circle((256, 400), 15)
                n_kps = n_agent_kps
            else:
                self.block = obj = self.add_tee((256, 300), 0)
                n_kps = n_block_kps

            self.screen = pygame.Surface((512, 512))
            self.screen.fill(pygame.Color("white"))
            draw_options = DrawOptions(self.screen)
            self.space.debug_draw(draw_options)
            # pygame.display.flip()
            img = np.uint8(pygame.surfarray.array3d(self.screen).transpose(1, 0, 2))
            obj_mask = (img != np.array([255, 255, 255], dtype=np.uint8)).any(axis=-1)

            tf_img_obj = cls.get_tf_img_obj(obj)
            xy_img = np.moveaxis(np.array(np.indices((512, 512))), 0, -1)[:, :, ::-1]
            local_coord_img = tf_img_obj.inverse(xy_img.reshape(-1, 2)).reshape(
                xy_img.shape
            )
            obj_local_coords = local_coord_img[obj_mask]

            # furthest point sampling
            init_idx = rng.choice(len(obj_local_coords))
            obj_local_kps = farthest_point_sampling(obj_local_coords, n_kps, init_idx)
            small_shift = rng.uniform(0, 1, size=obj_local_kps.shape)
            obj_local_kps += small_shift

            local_keypoint_map[name] = obj_local_kps

        return cls(local_keypoint_map=local_keypoint_map, **kwargs)

    @staticmethod
    def get_tf_img(pose: Sequence):
        pos = pose[:2]
        rot = pose[2]
        tf_img_obj = st.AffineTransform(translation=pos, rotation=rot)
        return tf_img_obj

    @classmethod
    def get_tf_img_obj(cls, obj: pymunk.Body):
        pose = tuple(obj.position) + (obj.angle,)
        return cls.get_tf_img(pose)

    def get_keypoints_global(
        self, pose_map: Dict[set, Union[Sequence, pymunk.Body]], is_obj=False
    ):
        kp_map = dict()
        for key, value in pose_map.items():
            if is_obj:
                tf_img_obj = self.get_tf_img_obj(value)
            else:
                tf_img_obj = self.get_tf_img(value)
            kp_local = self.local_keypoint_map[key]
            kp_global = tf_img_obj(kp_local)
            kp_map[key] = kp_global
        return kp_map

    def draw_keypoints(self, img, kps_map, radius=1):
        scale = np.array(img.shape[:2]) / np.array([512, 512])
        for key, value in kps_map.items():
            color = self.color_map[key].tolist()
            coords = (value * scale).astype(np.int32)
            for coord in coords:
                cv2.circle(img, coord, radius=radius, color=color, thickness=-1)
        return img

    def draw_keypoints_pose(self, img, pose_map, is_obj=False, **kwargs):
        kp_map = self.get_keypoints_global(pose_map, is_obj=is_obj)
        return self.draw_keypoints(img, kps_map=kp_map, **kwargs)


class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        """Draw a pymunk.Space on a pygame.Surface object.

        Typical usage::

        >>> import pymunk
        >>> surface = pygame.Surface((10,10))
        >>> space = pymunk.Space()
        >>> options = pymunk.pygame_util.DrawOptions(surface)
        >>> space.debug_draw(options)

        You can control the color of a shape by setting shape.color to the color
        you want it drawn in::

        >>> c = pymunk.Circle(None, 10)
        >>> c.color = pygame.Color("pink")

        See pygame_util.demo.py for a full example

        Since pygame uses a coordinate system where y points down (in contrast
        to many other cases), you either have to make the physics simulation
        with Pymunk also behave in that way, or flip everything when you draw.

        The easiest is probably to just make the simulation behave the same
        way as Pygame does. In that way all coordinates used are in the same
        orientation and easy to reason about::

        >>> space = pymunk.Space()
        >>> space.gravity = (0, -1000)
        >>> body = pymunk.Body()
        >>> body.position = (0, 0) # will be positioned in the top left corner
        >>> space.debug_draw(options)

        To flip the drawing its possible to set the module property
        :py:data:`positive_y_is_up` to True. Then the pygame drawing will flip
        the simulation upside down before drawing::

        >>> positive_y_is_up = True
        >>> body = pymunk.Body()
        >>> body.position = (0, 0)
        >>> # Body will be position in bottom left corner

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(
            self.surface, light_color(fill_color).as_int(), p, round(radius - 4), 0
        )

        circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        p2 = to_pygame(circle_edge, self.surface)
        line_r = 2 if radius > 20 else 1
        # pygame.draw.lines(self.surface, outline_color.as_int(), False, [p, p2], line_r)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]

        radius = 2
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)

        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def get_mouse_pos(surface: pygame.Surface) -> Tuple[int, int]:
    """Get position of the mouse pointer in pymunk coordinates."""
    p = pygame.mouse.get_pos()
    return from_pygame(p, surface)


def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates.

    Note that in case positive_y_is_up is False, this function won't actually do
    anything except converting the point to integers.
    """
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])


def from_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pygame surface local coordinates to
    pymunk coordinates
    """
    return to_pygame(p, surface)


def light_color(color: SpaceDebugColor):
    color = np.minimum(
        1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255])
    )
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class PushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_action=True,
        render_size=96,
        reset_to_state=None,
    ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["video.frames_per_second"]
        # legcay set_state for data compatibility
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float64),
            high=np.array([ws, ws, ws, ws, np.pi * 2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64,
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float64),
            high=np.array([ws, ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state
        self.coverage_arr = []

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array(
                [
                    rs.randint(50, 450),
                    rs.randint(50, 450),
                    rs.randint(100, 400),
                    rs.randint(100, 400),
                    rs.randn() * 2 * np.pi - np.pi,
                ]
            )
        self._set_state(state)

        self.coverage_arr = []
        observation = self._get_obs()
        return observation

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                    Vec2d(0, 0) - self.agent.velocity
                )
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = False  # coverage > self.success_threshold

        self.coverage_arr.append(coverage)
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(
                Vec2d(*pygame.mouse.get_pos()), self.screen
            )
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return TeleopAgent(act)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position)
            + tuple(self.block.position)
            + (self.block.angle % (2 * np.pi),)
        )
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [
                pymunk.pygame_util.to_pygame(
                    goal_body.local_to_world(v), draw_options.surface
                )
                for v in shape.get_vertices()
            ]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8 / 96 * self.render_size)
                thickness = int(1 / 96 * self.render_size)
                cv2.drawMarker(
                    img,
                    coord,
                    color=(255, 0, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size,
                    thickness=thickness,
                )
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], rotation=self.goal_pose[2]
        )
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2], rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(matrix=tf_img_obj.params @ tf_obj_new.params)
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0])
            + list(tf_img_new.translation)
            + [tf_img_new.rotation]
        )
        self._set_state(new_state)
        return new_state

    def set_task_goal(self, goal):
        self.goal_pose = goal

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = pygame.Color("LightGreen")
        self.goal_pose = np.array([256, 256, np.pi / 4])  # x, y, theta (in radians)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95  # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color(
            "LightGray"
        )  # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color("RoyalBlue")
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color("LightSlateGray")
        self.space.add(body, shape)
        return body

    def add_tee(
        self,
        position,
        angle,
        scale=30,
        color="LightSlateGray",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body


class PymunkKeypointManager:
    def __init__(
        self,
        local_keypoint_map: Dict[str, np.ndarray],
        color_map: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        local_keypoint_map:
            "<attribute_name>": (N,2) floats in object local coordinate
        """
        if color_map is None:
            cmap = cm.get_cmap("tab10")
            color_map = dict()
            for i, key in enumerate(local_keypoint_map.keys()):
                color_map[key] = (np.array(cmap.colors[i]) * 255).astype(np.uint8)

        self.local_keypoint_map = local_keypoint_map
        self.color_map = color_map

    @property
    def kwargs(self):
        return {
            "local_keypoint_map": self.local_keypoint_map,
            "color_map": self.color_map,
        }

    @classmethod
    def create_from_pusht_env(cls, env, n_block_kps=9, n_agent_kps=3, seed=0, **kwargs):
        rng = np.random.default_rng(seed=seed)
        local_keypoint_map = dict()
        for name in ["block", "agent"]:
            self = env
            self.space = pymunk.Space()
            if name == "agent":
                self.agent = obj = self.add_circle((256, 400), 15)
                n_kps = n_agent_kps
            else:
                self.block = obj = self.add_tee((256, 300), 0)
                n_kps = n_block_kps

            self.screen = pygame.Surface((512, 512))
            self.screen.fill(pygame.Color("white"))
            draw_options = DrawOptions(self.screen)
            self.space.debug_draw(draw_options)
            # pygame.display.flip()
            img = np.uint8(pygame.surfarray.array3d(self.screen).transpose(1, 0, 2))
            obj_mask = (img != np.array([255, 255, 255], dtype=np.uint8)).any(axis=-1)

            tf_img_obj = cls.get_tf_img_obj(obj)
            xy_img = np.moveaxis(np.array(np.indices((512, 512))), 0, -1)[:, :, ::-1]
            local_coord_img = tf_img_obj.inverse(xy_img.reshape(-1, 2)).reshape(
                xy_img.shape
            )
            obj_local_coords = local_coord_img[obj_mask]

            # furthest point sampling
            init_idx = rng.choice(len(obj_local_coords))
            obj_local_kps = farthest_point_sampling(obj_local_coords, n_kps, init_idx)
            small_shift = rng.uniform(0, 1, size=obj_local_kps.shape)
            obj_local_kps += small_shift

            local_keypoint_map[name] = obj_local_kps

        return cls(local_keypoint_map=local_keypoint_map, **kwargs)

    @staticmethod
    def get_tf_img(pose: Sequence):
        pos = pose[:2]
        rot = pose[2]
        tf_img_obj = st.AffineTransform(translation=pos, rotation=rot)
        return tf_img_obj

    @classmethod
    def get_tf_img_obj(cls, obj: pymunk.Body):
        pose = tuple(obj.position) + (obj.angle,)
        return cls.get_tf_img(pose)

    def get_keypoints_global(
        self, pose_map: Dict[set, Union[Sequence, pymunk.Body]], is_obj=False
    ):
        kp_map = dict()
        for key, value in pose_map.items():
            if is_obj:
                tf_img_obj = self.get_tf_img_obj(value)
            else:
                tf_img_obj = self.get_tf_img(value)
            kp_local = self.local_keypoint_map[key]
            kp_global = tf_img_obj(kp_local)
            kp_map[key] = kp_global
        return kp_map

    def draw_keypoints(self, img, kps_map, radius=1):
        scale = np.array(img.shape[:2]) / np.array([512, 512])
        for key, value in kps_map.items():
            color = self.color_map[key].tolist()
            coords = (value * scale).astype(np.int32)
            for coord in coords:
                cv2.circle(img, coord, radius=radius, color=color, thickness=-1)
        return img

    def draw_keypoints_pose(self, img, pose_map, is_obj=False, **kwargs):
        kp_map = self.get_keypoints_global(pose_map, is_obj=is_obj)
        return self.draw_keypoints(img, kps_map=kp_map, **kwargs)


class PushTKeypointsEnv(PushTEnv):
    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_size=96,
        keypoint_visible_rate=1.0,
        agent_keypoints=False,
        draw_keypoints=False,
        reset_to_state=None,
        render_action=False,
        local_keypoint_map: Dict[str, np.ndarray] = None,
        color_map: Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__(
            legacy=legacy,
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            render_action=render_action,
        )
        ws = self.window_size

        if local_keypoint_map is None:
            # create default keypoint definition
            kp_kwargs = self.genenerate_keypoint_manager_params()
            local_keypoint_map = kp_kwargs["local_keypoint_map"]
            color_map = kp_kwargs["color_map"]

        # create observation spaces
        Dblockkps = np.prod(local_keypoint_map["block"].shape)
        Dagentkps = np.prod(local_keypoint_map["agent"].shape)
        Dagentpos = 2

        Do = Dblockkps
        if agent_keypoints:
            # blockkp + agnet_pos
            Do += Dagentkps
        else:
            # blockkp + agnet_kp
            Do += Dagentpos
        # obs + obs_mask
        Dobs = Do * 2

        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        # mask range 0-1
        high[Do:] = 1.0

        # (block_kps+agent_kps, xy+confidence)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=low.shape, dtype=np.float64
        )

        self.keypoint_visible_rate = keypoint_visible_rate
        self.agent_keypoints = agent_keypoints
        self.draw_keypoints = draw_keypoints
        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map, color_map=color_map
        )
        self.draw_kp_map = None

    @classmethod
    def genenerate_keypoint_manager_params(cls):
        env = PushTEnv()
        kp_manager = PymunkKeypointManager.create_from_pusht_env(env)
        kp_kwargs = kp_manager.kwargs
        return kp_kwargs

    def _get_obs(self):
        # get keypoints
        obj_map = {"block": self.block}
        if self.agent_keypoints:
            obj_map["agent"] = self.agent

        kp_map = self.kp_manager.get_keypoints_global(pose_map=obj_map, is_obj=True)
        # python dict guerentee order of keys and values
        kps = np.concatenate(list(kp_map.values()), axis=0)

        # select keypoints to drop
        n_kps = kps.shape[0]
        visible_kps = self.np_random.random(size=(n_kps,)) < self.keypoint_visible_rate
        kps_mask = np.repeat(visible_kps[:, None], 2, axis=1)

        # save keypoints for rendering
        vis_kps = kps.copy()
        vis_kps[~visible_kps] = 0
        draw_kp_map = {"block": vis_kps[: len(kp_map["block"])]}
        if self.agent_keypoints:
            draw_kp_map["agent"] = vis_kps[len(kp_map["block"]) :]
        self.draw_kp_map = draw_kp_map

        # construct obs
        obs = kps.flatten()
        obs_mask = kps_mask.flatten()
        if not self.agent_keypoints:
            # passing agent position when keypoints are not available
            agent_pos = np.array(self.agent.position)
            obs = np.concatenate([obs, agent_pos])
            obs_mask = np.concatenate([obs_mask, np.ones((2,), dtype=bool)])

        # obs, obs_mask
        obs = np.concatenate([obs, obs_mask.astype(obs.dtype)], axis=0)
        return obs

    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        if self.draw_keypoints:
            self.kp_manager.draw_keypoints(
                img, self.draw_kp_map, radius=int(img.shape[0] / 96)
            )
        return img


import zarr
from pathlib import Path


class Normalizer:
    def __init__(self, data_directory, device="cuda", onehot_goals=False):
        data_directory = Path(data_directory)
        src_root = zarr.group(data_directory / "pusht_cchi_v7_replay.zarr")

        # numpy backend
        meta = dict()
        for key, value in src_root["meta"].items():
            if len(value.shape) == 0:
                meta[key] = np.array(value)
            else:
                meta[key] = value[:]

        keys = src_root["data"].keys()
        data = dict()
        for key in keys:
            arr = src_root["data"][key]
            data[key] = arr[:]
        # meta['episode_ends'] = meta['episode_ends'][:int(meta['episode_ends'].shape[0]/4)]
        # data['obs'] = data['obs'][:meta['episode_ends'][-1]]
        # data['action'] = data['action'][:meta['episode_ends'][-1]]

        observations = []
        actions = []
        masks = []
        start = 0
        agent_pos = data["state"][:, :2]
        keypoint_obs = np.concatenate(
            [data["keypoint"].reshape(data["keypoint"].shape[0], -1), agent_pos],
            axis=-1,
        )

        for end in meta["episode_ends"]:
            if (300 - (end - start)) <= 0:
                print("too small capacity")
            observations.append(
                np.concatenate(
                    (keypoint_obs[start:end], np.zeros((300 - (end - start), 20)))
                )
            )
            actions.append(
                np.concatenate(
                    (data["action"][start:end], np.zeros((300 - (end - start), 2)))
                )
            )
            masks.append(
                np.concatenate(
                    (np.ones((end - start)), np.zeros((300 - (end - start))))
                )
            )

            start = end
        observations = np.array(observations)
        actions = np.array(actions)
        masks = np.array(masks)

        self.observations_stats = self.get_data_stats(observations)

    def get_data_stats(self, data):
        data = data.reshape(-1, data.shape[-1])
        stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
        return stats

    def normalize_data(self, data, stats):
        # nomalize to [0,1]
        ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        return ndata

    def unnormalize_data(self, ndata, stats):
        ndata = (ndata + 1) / 2
        data = ndata * (stats["max"] - stats["min"]) + stats["min"]
        return data


class PushWrapper(gym.Wrapper):
    def __init__(self, env, id, data_directory, visual_input=False):
        super(PushWrapper, self).__init__(env)
        self.env = env
        if not visual_input:
            self.normalizer = (Normalizer(data_directory, onehot_goals=False),)
        self.id = id
        self.visual_input = visual_input

    def reset(self, *args, **kwargs):
        print("reset env!!!!!")
        obs = self.env.reset(*args, **kwargs)
        self.step_idx = 0
        if self.visual_input:
            return_obs = self.env.render(mode="rgb_array")
        else:
            return_obs = self.normalizer[0].normalize_data(
                obs[:20], self.normalizer[0].observations_stats
            )
        return return_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["image"] = self.env.render(mode="rgb_array")
        info["all_completions_ids"] = []
        if self.step_idx > 300:
            done = True
        else:
            done = False
        self.step_idx += 1
        info["max_coverage"] = max(self.coverage_arr)
        info["final_coverage"] = self.coverage_arr[-1]
        if self.visual_input:
            return_obs = info["image"]
        else:
            return_obs = self.normalizer[0].normalize_data(
                obs[:20], self.normalizer[0].observations_stats
            )
        return return_obs, reward, done, info


def get_goal_fn(
    data_directory: str,
    goal_conditional: Optional[str] = None,
    seed: Optional[int] = None,
    train_fraction: Optional[float] = None,
    visual_input: bool = False,
    goal_seq_len: Optional[int] = None,
    goal_dim=20,
):
    if goal_dim > 0:
        if visual_input:
            relay_traj = PushTSequenceDataset(
                data_directory,
                onehot_goals=(goal_conditional == "onehot"),
                visual_input=visual_input,
                device="cpu",
            )
            train_idx, val_idx = get_split_idx(
                len(relay_traj),
                seed=seed,
                train_fraction=train_fraction or 1.0,
            )
            goal_fn = lambda env, state, goal_idx, frame_idx: None

            def future_goal_fn(env, state, goal_idx, frame_idx):  # type: ignore
                obs, _, goal = relay_traj[train_idx[goal_idx]]
                goal_start_idx = np.random.randint(0, len(goal) - goal_seq_len + 1)
                return_state_goal = goal[
                    goal_start_idx : goal_start_idx + goal_seq_len
                ].clone()
                return_visual_goal = obs[
                    goal_start_idx : goal_start_idx + goal_seq_len
                ].clone()
                env.set_task_goal(return_state_goal[-1, :3])
                # return_goal = torch.concat((return_goal, torch.zeros((goal_seq_len, goal_dim - 3)).cuda()), dim=1)
                return return_visual_goal, None

            goal_fn = future_goal_fn
        else:
            relay_traj = PushTSequenceDataset(
                data_directory,
                onehot_goals=(goal_conditional == "onehot"),
                visual_input=visual_input,
                device="cpu",
            )
            train_idx, val_idx = get_split_idx(
                len(relay_traj),
                seed=seed,
                train_fraction=train_fraction or 1.0,
            )
            goal_fn = lambda env, state, goal_idx, frame_idx: None

            def future_goal_fn(env, state, goal_idx, frame_idx):  # type: ignore
                _, _, goal = relay_traj[train_idx[goal_idx]]
                goal_start_idx = np.random.randint(0, len(goal) - goal_seq_len + 1)
                return_goal = goal[
                    goal_start_idx : goal_start_idx + goal_seq_len
                ].clone()
                env.set_task_goal(return_goal[-1, :3])
                # return_goal = torch.concat((return_goal, torch.zeros((goal_seq_len, goal_dim - 3)).cuda()), dim=1)
                return return_goal, None

            goal_fn = future_goal_fn
    else:

        def dummy_goal_fn(env, state, goal_idx, frame_idx):  # type: ignore
            return np.zeros((3, 20)), None

        goal_fn = dummy_goal_fn

    return goal_fn


def get_split_idx(l, seed, train_fraction=0.95):
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(l, generator=rng).tolist()
    l_train = int(l * train_fraction)
    return idx[:l_train], idx[l_train:]
