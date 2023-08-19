import roar_py_carla
import roar_py_interface
import carla
import numpy as np
import asyncio
from typing import Optional, Dict, Any, List
import pygame
from PIL import Image
import transforms3d
from manual_control import ManualControlViewer


def normalize_rad(rad: float):
    return (rad + np.pi) % (2 * np.pi) - np.pi


def filter_waypoints(location: np.ndarray, current_idx: int, waypoints: List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint: roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )

    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i % len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx


def load_turn_waypoints(file_path: str) -> List[int]:
    with open(file_path, 'r') as file:
        return [int(line.strip()) for line in file]
SLOW = "slow"
MEDIUMSLOW = "mediumslow"
MEDIUMFAST = "mediumfast"
FAST = "fast"

async def main():
    carla_client = carla.Client('127.0.0.1', 2000)
    carla_client.set_timeout(5.0)
    roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
    manual_viewer = ManualControlViewer()

    carla_world = roar_py_instance.world
    carla_world.set_asynchronous(True)
    carla_world.set_control_steps(0.0, 0.01)

    way_points = carla_world.maneuverable_waypoints
    vehicle = carla_world.spawn_vehicle(
        "vehicle.audi.a2",
        way_points[10].location + np.array([0, 0, 1]),
        way_points[10].roll_pitch_yaw
    )

    current_waypoint_idx = 10

    assert vehicle is not None
    camera = vehicle.attach_camera_sensor(
        roar_py_interface.RoarPyCameraSensorDataRGB,
        np.array([-2.0 * vehicle.bounding_box.extent[0], 0.0, 3.0 * vehicle.bounding_box.extent[2]]),
        np.array([0, 10 / 180.0 * np.pi, 0]),
        image_width=1024,
        image_height=768
    )
    assert camera is not None

    start_time = carla_world.last_tick_elapsed_seconds

    # Load the braking waypoints
    turn_waypoints = load_turn_waypoints("turns.txt")
    slowcount = 0
    current_speed_state = FAST
    try:
        while True:
            await carla_world.step()
            vehicle_location = vehicle.get_3d_location()
            vehicle_rotation = vehicle.get_roll_pitch_yaw()

            camera_data = await camera.receive_observation()
            render_ret = manual_viewer.render(camera_data)
            if render_ret is None:
                break

            current_waypoint_idx = filter_waypoints(
                vehicle_location,
                current_waypoint_idx,
                way_points
            )
            #print(f"Current Waypoint Index: {current_waypoint_idx}")
            

            waypoint_to_follow = way_points[(current_waypoint_idx + 3) % len(way_points)]

            vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
            heading_to_waypoint = np.arctan2(vector_to_waypoint[1], vector_to_waypoint[0])
            delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])
            #print(delta_heading*180/np.pi)

            steer_control = (
                -20.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
            ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
            steer_control = np.clip(steer_control, -1.0, 1.0)
            throttle_control = 0.5 * (30 - np.linalg.norm(vehicle.get_linear_3d_velocity()))
            #print(np.clip(throttle_control, 0.0, 1.0))

            if current_waypoint_idx >15 and current_waypoint_idx < 160:
                
                if current_waypoint_idx+2 in turn_waypoints or current_waypoint_idx+1 in turn_waypoints or current_waypoint_idx in turn_waypoints:
                    steer_control = (
                    -10.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
                    ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
                    steer_control = np.clip(steer_control, -1.0, 1.0)
                    control = {
                        "throttle": 0,
                        "steer": steer_control,
                        "brake": 1.0,
                        "hand_brake": 0.0,
                        "reverse": 0,
                        "target_gear": 0
                    }
                    print("brake")
                
                
                else:
                    throttle_control = 0.05 * (35 - np.linalg.norm(vehicle.get_linear_3d_velocity()))  
                    control = {
                        "throttle": np.clip(throttle_control, 0.0, 1.0),
                        "steer": steer_control,
                        "brake": np.clip(-throttle_control, 0.0, 1.0),
                        "hand_brake": 0.0,
                        "reverse": 0,
                        "target_gear": 0
                    }

            elif current_waypoint_idx >160 and current_waypoint_idx < 920:
                if current_waypoint_idx+22 in turn_waypoints or current_waypoint_idx+21 in turn_waypoints or current_waypoint_idx+ 20 in turn_waypoints or current_waypoint_idx+19 in turn_waypoints or current_waypoint_idx+18 in turn_waypoints or current_waypoint_idx+17 in turn_waypoints or current_waypoint_idx+16 in turn_waypoints or current_waypoint_idx+15 in turn_waypoints or current_waypoint_idx+14 in turn_waypoints or current_waypoint_idx+13 in turn_waypoints or current_waypoint_idx+12 in turn_waypoints or current_waypoint_idx+11 in turn_waypoints or current_waypoint_idx+10 in turn_waypoints or current_waypoint_idx+9 in turn_waypoints or current_waypoint_idx+8 in turn_waypoints or current_waypoint_idx+7 in turn_waypoints or current_waypoint_idx+6 in turn_waypoints or current_waypoint_idx+5 in turn_waypoints or current_waypoint_idx+4 in turn_waypoints or current_waypoint_idx+3 in turn_waypoints or current_waypoint_idx+2 in turn_waypoints or current_waypoint_idx+1 in turn_waypoints or current_waypoint_idx in turn_waypoints:
                    steer_control = (
                    -10.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
                    ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
                    steer_control = np.clip(steer_control, -1.0, 1.0)
                    control = {
                        "throttle": 0,
                        "steer": steer_control,
                        "brake": 1.0,
                        "hand_brake": 0.0,
                        "reverse": 0,
                        "target_gear": 0
                    }
                    #print("brake")
                else:
                    throttle_control = 0.5 * (50 - np.linalg.norm(vehicle.get_linear_3d_velocity()))
                    steer_control = (
                    -40.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
                    ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
                    steer_control = np.clip(steer_control, -1.0, 1.0)
                
                    control = {
                        "throttle": np.clip(throttle_control, 0.0, 1.0),
                        "steer": steer_control,
                        "brake": np.clip(-throttle_control, 0.0, 1.0),
                        "hand_brake": 0.0,
                        "reverse": 0,
                        "target_gear": 0
                    }
            
            elif current_waypoint_idx >921 and current_waypoint_idx < 1445:
                if current_waypoint_idx+14 in turn_waypoints or current_waypoint_idx+13 in turn_waypoints or current_waypoint_idx+12 in turn_waypoints or current_waypoint_idx+11 in turn_waypoints or current_waypoint_idx+10 in turn_waypoints or current_waypoint_idx+9 in turn_waypoints or current_waypoint_idx+8 in turn_waypoints or current_waypoint_idx+7 in turn_waypoints or current_waypoint_idx+6 in turn_waypoints or current_waypoint_idx+5 in turn_waypoints or current_waypoint_idx+4 in turn_waypoints or current_waypoint_idx+3 in turn_waypoints or current_waypoint_idx+2 in turn_waypoints or current_waypoint_idx+1 in turn_waypoints or current_waypoint_idx in turn_waypoints:
                    steer_control = (
                    -10.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
                    ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
                    steer_control = np.clip(steer_control, -1.0, 1.0)
                    control = {
                        "throttle": 0,
                        "steer": steer_control,
                        "brake": 1.0,
                        "hand_brake": 0.0,
                        "reverse": 0,
                        "target_gear": 0
                    }
                    #print("brake")
                else:
                    throttle_control = 0.5 * (35 - np.linalg.norm(vehicle.get_linear_3d_velocity()))
                    steer_control = (
                    -20.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
                    ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
                    steer_control = np.clip(steer_control, -1.0, 1.0)
                
                    control = {
                        "throttle": np.clip(throttle_control, 0.0, 1.0),
                        "steer": steer_control,
                        "brake": np.clip(-throttle_control, 0.0, 1.0),
                        "hand_brake": 0.0,
                        "reverse": 0,
                        "target_gear": 0
                    }

            elif current_waypoint_idx >1446 and current_waypoint_idx < 6250:
                
                if  current_waypoint_idx+10 in turn_waypoints or current_waypoint_idx+9 in turn_waypoints or current_waypoint_idx+8 in turn_waypoints or current_waypoint_idx+7 in turn_waypoints or current_waypoint_idx+6 in turn_waypoints or current_waypoint_idx+5 in turn_waypoints or current_waypoint_idx+4 in turn_waypoints or current_waypoint_idx+3 in turn_waypoints or current_waypoint_idx+2 in turn_waypoints or current_waypoint_idx+1 in turn_waypoints:
                    steer_control = (
                    -10.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
                    ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
                    steer_control = np.clip(steer_control, -1.0, 1.0)
                    control = {
                        "throttle": 0,
                        "steer": steer_control,
                        "brake": 1.0,
                        "hand_brake": 0.0,
                        "reverse": 0,
                        "target_gear": 0
                    }
                    #print("brake")

                waypoint_to_turn = way_points[(current_waypoint_idx + 3) % len(way_points)]


                # Calculate delta vector towards the target waypoint
                vector_to_turn = (waypoint_to_turn.location - vehicle_location)[:2]
                heading_to_turn = np.arctan2(vector_to_turn[1],vector_to_turn[0])
                delta_turn = normalize_rad(heading_to_turn - vehicle_rotation[2])

                drive_angle = delta_turn*180/np.pi

            
                if current_speed_state == SLOW:
                    if drive_angle >=-4.5 and drive_angle <= 4.5:
                        next_speed_state = MEDIUMFAST
                    else:
                        next_speed_state = SLOW


                elif current_speed_state == FAST:
                    if drive_angle <= -2.5 or drive_angle >=2.5:
                        next_speed_state = MEDIUMSLOW
                    else:
                        next_speed_state = FAST


                elif current_speed_state == MEDIUMFAST:
                    if drive_angle < -4.5 or drive_angle>4.5:
                        next_speed_state = MEDIUMSLOW
                    elif drive_angle >=-2.5 and drive_angle <=2.5:
                        next_speed_state = FAST
                    else:
                        next_speed_state = MEDIUMFAST


                elif current_speed_state == MEDIUMSLOW:
                    #if drive_angle > -6 and drive_angle < 6:
                        #next_speed_state = SLOW 
                    #print(f"Medium A Slow {drive_angle}")
                    #print(slowcount)


                    if slowcount >= 3:
                        next_speed_state = MEDIUMFAST
                        slowcount = 0
                    elif drive_angle >= -5.5 and drive_angle <= 5.5:
                        next_speed_state = MEDIUMFAST
                    
                    else:
                        next_speed_state = MEDIUMSLOW
            
                #fast
                if next_speed_state == FAST:
                    steer_control = (
                    -2.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
                    ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
                    steer_control = np.clip(steer_control, -1.0, 1.0)




                    throttle_control = 0.1 * (34 - np.linalg.norm(vehicle.get_linear_3d_velocity()))


                    control = {
                        "throttle": np.clip(throttle_control, 0.0, 1.0),
                        "steer": steer_control,
                        "brake": np.clip(-throttle_control, 0.0, 1.0),
                        "hand_brake": 0.0,
                        "reverse": 0,
                        "target_gear": 0
                    
                    }
                    #print(f"Fast {drive_angle}")
                    #medium 
                elif next_speed_state == MEDIUMFAST:
                    steer_control = (
                    -8.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
                    ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
                    steer_control = np.clip(steer_control, -1.0, 1.0)




                    throttle_control = 0.05 * (33 - np.linalg.norm(vehicle.get_linear_3d_velocity()))


                    control = {
                        "throttle": np.clip(throttle_control, 0.0, 1.0),
                        "steer": steer_control,
                        "brake": np.clip(-throttle_control, 0.0, 1.0),
                        "hand_brake": 0.0,
                        "reverse": 0,
                        "target_gear": 0
                    
                    }
                    #print(f"Mediumfast {drive_angle}")
                #Mediumslow
                elif next_speed_state == MEDIUMSLOW:
                    slowcount += 1
                    steer_control = (
                    -8.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
                    ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
                    steer_control = np.clip(steer_control, -1.0, 1.0)




                    throttle_control = 0.05 * (15 - np.linalg.norm(vehicle.get_linear_3d_velocity()))


                    control = {
                        "throttle": np.clip(throttle_control, 0.0, 1.0),
                        "steer": steer_control,
                        "brake": 0.09,
                        "hand_brake": 0.0,
                        "reverse": 0,
                        "target_gear": 0
                    
                    }
                    #print(f"Medium Slow {drive_angle}")
                    #print(slowcount)


                    #slow
                elif next_speed_state == SLOW:
                    steer_control = (
                    -   8.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
                    ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
                    steer_control = np.clip(steer_control, -1.0, 1.0)


                    # Proportional controller to control the vehicle's speed towards 40 m/s
                    throttle_control = 0.05 * (20 - np.linalg.norm(vehicle.get_linear_3d_velocity()))


                    control = {
                        "throttle": np.clip(throttle_control, 0.0, 1.0),
                        "steer": steer_control,
                        "brake": np.clip(-throttle_control, 0.0, 1.0),
                        "hand_brake": 0.0,
                        "reverse": 0,
                        "target_gear": 0
                    }
                    #print(f"Slow{drive_angle}")
            
                current_speed_state = next_speed_state




                await vehicle.apply_action(control)
            
            elif current_waypoint_idx >6251 and current_waypoint_idx < 7000:
                if current_waypoint_idx+14 in turn_waypoints or current_waypoint_idx+13 in turn_waypoints or current_waypoint_idx+12 in turn_waypoints or current_waypoint_idx+11 in turn_waypoints or current_waypoint_idx+10 in turn_waypoints or current_waypoint_idx+9 in turn_waypoints or current_waypoint_idx+8 in turn_waypoints or current_waypoint_idx+7 in turn_waypoints or current_waypoint_idx+6 in turn_waypoints or current_waypoint_idx+5 in turn_waypoints or current_waypoint_idx+4 in turn_waypoints or current_waypoint_idx+3 in turn_waypoints or current_waypoint_idx+2 in turn_waypoints or current_waypoint_idx+1 in turn_waypoints or current_waypoint_idx in turn_waypoints:
                    steer_control = (
                    -10.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
                    ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
                    steer_control = np.clip(steer_control, -1.0, 1.0)
                    control = {
                        "throttle": 0,
                        "steer": steer_control,
                        "brake": 1.0,
                        "hand_brake": 0.0,
                        "reverse": 0,
                        "target_gear": 0
                    }
                    #print("brake")
                else:
                    throttle_control = 0.5 * (35 - np.linalg.norm(vehicle.get_linear_3d_velocity()))
                    steer_control = (
                    -20.0 / np.sqrt(np.linalg.norm(vehicle.get_linear_3d_velocity())) * delta_heading / np.pi
                    ) if np.linalg.norm(vehicle.get_linear_3d_velocity()) > 1e-2 else -np.sign(delta_heading)
                    steer_control = np.clip(steer_control, -1.0, 1.0)
                
                    control = {
                        "throttle": np.clip(throttle_control, 0.0, 1.0),
                        "steer": steer_control,
                        "brake": np.clip(-throttle_control, 0.0, 1.0),
                        "hand_brake": 0.0,
                        "reverse": 0,
                        "target_gear": 0
                    }


                
            


            else:
                control = {
                    "throttle": np.clip(throttle_control, 0.0, 1.0),
                    "steer": steer_control,
                    "brake": np.clip(-throttle_control, 0.0, 1.0),
                    "hand_brake": 0.0,
                    "reverse": 0,
                    "target_gear": 0
                }
            await vehicle.apply_action(control)

        

    finally:
        roar_py_instance.close()
        end_time = carla_world.last_tick_elapsed_seconds
        delta_time = end_time - start_time
        print(f"Total time: {delta_time} seconds")
        #print(current_waypoint_idx)


if __name__ == '__main__':
    asyncio.run(main())
