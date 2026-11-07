import gymnasium as gym
import highway_env
import numpy as np
from highway_env.vehicle.behavior import IDMVehicle

class Intent(IDMVehicle):
    def act(self, action=None):
        if getattr(self, 'elapsed', 0.0) >= getattr(self, 'brake_at', float('inf')):
            self.target_speed = self.brake_speed
        super().act(action)
    
    def step(self, dt):
        self.elapsed = getattr(self, 'elapsed', 0.0) + dt
        super().step(dt)
    
    @property
    def intent(self):
        front, rear = self.road.neighbour_vehicles(self, self.lane_index)
        acceleration = self.acceleration(self, front, rear)
        return 'brake' if acceleration < -0.5 else 'accelerate' if acceleration > 0.5 else 'maintain'

class Truck(Intent):
    LENGTH = 12.0
    WIDTH = 2.5

class DrivingEnv:
    def __init__(self, bits_price=0.01, seed=11, scenario=None, render_mode=None):
            if scenario not in (None, 'lead_braking', 'truck_occlusion', 'mixed'):
                raise ValueError('Unknown scenario')
            self.scenario_choice = scenario
            self.bits_price = bits_price
            self.seed = seed
            self.env = gym.make('highway-v0', render_mode=render_mode, config={'lanes_count': 3,
                'vehicles_count': 0, 'duration': 20, 'initial_lane_id': 1, 'policy_frequency': 2,
                'observation': {'type': 'LidarObservation', 'cells': 16,
                                'maximum_range': 25, 'normalize': True}})
            self.requests = self.bits = 0
            self.pending = False
    
    def reset(self, seed=None):
            obs, info = self.env.reset(seed=self.seed if seed is None else seed)
            self.requests = self.bits = 0
            self.pending = False
            self.steps = 0
            self.target_lane = 2
            rng = np.random.default_rng(self.seed if seed is None else seed)
            base = self.env.unwrapped
            ego = base.vehicle
            ego.position = np.array([100.0, 4.0]); ego.speed = 25.0
            ego.target_speed = 25.0; ego.heading = 0.0
            base.road.vehicles = [ego]
            self.scenario = self.scenario_choice or str(rng.choice(['lead_braking', 'truck_occlusion', 'mixed']))
            scenarios = [
                (1, rng.uniform(16, 20) if self.scenario == 'truck_occlusion' else rng.uniform(45, 60), rng.uniform(23, 26)),
                (2, rng.uniform(22, 28), rng.uniform(23, 27)),
                (2, rng.uniform(-25, -18), rng.uniform(24, 28)),
                (2, rng.uniform(55, 65), rng.uniform(24, 27)),
            ]
            if self.scenario == 'truck_occlusion':
                scenarios.append((1, 34.0, rng.uniform(22, 25)))
            for lane_index in range(3):
                for dx in (-95.0, -60.0, 95.0, 140.0):
                    if rng.random() < 0.45:
                        scenarios.append((lane_index, dx + rng.uniform(-4, 4), rng.uniform(20, 30)))
            for idx, (lane_index, dx, speed) in enumerate(scenarios):
                cls = Truck if idx == 0 and self.scenario == 'truck_occlusion' else Intent
                v = cls(base.road, np.array([100.0 + dx, 4.0 * lane_index]),
                               speed=float(speed), enable_lane_change=False)
                desired_change = float(rng.choice([0.0, 5.0, -5.0]))
                v.target_speed = max(5.0, v.speed + desired_change)
                if idx == 3 and self.scenario == 'lead_braking':
                    v.target_speed = v.speed
                    v.brake_at = float(rng.uniform(1.0, 2.0))
                    v.brake_speed = 8.0
                base.road.vehicles.append(v)
            self.episode_done = False
            info.update(scenario=self.scenario, vehicle_count=len(base.road.vehicles))
            return self.observation(self.lidar(ego, 25.0)), info

    def step(self, driving_action, reply):
        raise NotImplementedError("Not implemented yet")

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()
