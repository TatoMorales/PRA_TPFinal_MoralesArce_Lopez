# TP Final – Parte A

Este paquete contiene los nodos necesarios para resolver la Parte A del TP Final de **Principios de la Robótica Autónoma**: la navegación reactiva básica (`obstacle_navigation_node`) y la generación de mapa vía FastSLAM (`fast_slam_node`).

## Requisitos

- Ubuntu 22.04 + ROS 2 Humble.
- Paquetes de simulación: `turtlebot3_custom_simulation`, `turtlebot3_teleop` y los modelos de TurtleBot3 ya instalados y compilados.

## Compilación

```bash
colcon build --packages-select tp_final_package
source install/setup.bash
```

## Ejecución

1. **Simulación (Terminal 1):** en una terminal aparte lanzar el entorno de la casa indicado en la consigna:

   ```bash
   ros2 launch turtlebot3_custom_simulation custom_casa.launch.py
   ```

   Cambiar por otro launch (`custom_room.launch.py`, etc.) solo si se desea usar escenarios alternativos.

2. **Navegación reactiva (Terminal 2):** en lugar de teleoperar manualmente, lanzar el nodo que avanza y esquiva obstáculos automáticamente (puede usarse `turtlebot3_teleop` si se prefiere control manual):

   ```bash
   ros2 run tp_final_package obstacle_navigation_node --ros-args -p robot_model:=turtlebot3
   ```

   Ajustar `robot_model` a `turtlebot4` si se utiliza ese modelo.

3. **FastSLAM + RViz (Terminal 3):** en otra terminal (recordar `source install/setup.bash`) ejecutar

   ```bash
   ros2 launch tp_final_package fast_slam_bringup.launch.py
   ```

   El launch arranca el nodo `fast_slam_node` *y* abre RViz con la configuración pedida en la consigna (visualiza `/scan`, `/odom`, `/calc_odom`, `/estimated_pose` y `/map`). Si se desea otra vista, se puede pasar un archivo diferente vía `rviz_config:=</ruta/a/config.rviz>`.

4. **Guardado del mapa (Terminal 4):** cuando el recorrido cubra todo el entorno y el mapa esté limpio, abrir otra terminal (también con `source install/setup.bash`) y ejecutar:

   ```bash
   ros2 run nav2_map_server map_saver_cli -f ~/Documentos/Robotica/ws/src/PRA_TPFinal_MoralesArce_Lopez/maps/parte_a
   ```

   Cambiar la ruta del `-f` por la ubicación deseada; esto genera `parte_a.yaml` y `parte_a.pgm` con el mapa final.

## Localización (base para Parte B)

1. **Map server + fast_slam_localization:** en una terminal con el workspace sourceado ejecutar

   ```bash
   ros2 launch tp_final_package localization_bringup.launch.py
   ```

   Esto levanta `nav2_map_server` usando `maps/parte_a.yaml` y el nodo `fast_slam_localization_node`.

2. **Inicialización en RViz:** abrir RViz (puede ser la misma config de la Parte A), seleccionar la herramienta **2D Pose Estimate** y fijar la pose aproximada del robot; el nodo reinicializa la nube de partículas alrededor de esa pose.

3. **Objetivos:** con el mapa y la localización corriendo, cada vez que se use **2D Goal Pose** se publicará en `/goal_pose`, listo para que el planificador (`a_estrella_node`) lo consuma.

## Parte B – Navegación autónoma

1. **Simulación:** en una terminal aparte lanzar el entorno deseado (como en la Parte A).

   ```bash
   ros2 launch turtlebot3_custom_simulation custom_casa.launch.py
   ```

2. **Bring-up completo:** en otra terminal (con `source install/setup.bash`) correr

   ```bash
   ros2 launch tp_final_package part_b_bringup.launch.py
   ```

   Esto arranca `nav2_map_server` con `maps/parte_a.yaml`, `fast_slam_localización_node`, `a_estrella_node` y RViz. Si se desea lanzar también la simulación desde este mismo launch, agregar `launch_world:=true world_launch:=custom_casa_obs.launch.py`.

3. **Inicializar:** en RViz usar **2D Pose Estimate** para fijar la pose inicial del TurtleBot3.

4. **Planificar y ejecutar:** seleccionar el objetivo con **2D Goal Pose**. El nodo `a_estrella_node` planifica el camino, lo publica en `/planned_path`, lo sigue con Pure Pursuit, alinea el yaw final y replanifica automáticamente si detecta obstáculos no mapeados.

5. **Cambiar objetivo:** se puede publicar un nuevo goal en cualquier momento; la FSM entrará en estado de replanteo y seguirá el nuevo camino.

## Otros nodos

- `obstacle_navigation_node`: avanza a velocidad configurada (parámetro `linear_speed`, default 0.08 m/s) y rota 140° con velocidad `angular_speed` (default 0.4 rad/s) cuando el LIDAR detecta un obstáculo a 0.35 m o menos. Se selecciona TurtleBot3 o TurtleBot4 con `robot_model`.
