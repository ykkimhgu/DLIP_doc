# Simulator List

## Choosing a Simulator

&lt; Reference: [https://autodrive.readthedocs.io/en/latest/chapters/simulator/comparison.html](https://autodrive.readthedocs.io/en/latest/chapters/simulator/comparison.html) &gt;

### Comparison

![../../\_images/simulators.gif](https://autodrive.readthedocs.io/en/latest/_images/simulators.gif)

Comparison of different simulators:

<table>
  <thead>
    <tr>
      <th style="text-align:left">Table 1: Comparison of Autonomous Simulators</th>
      <th style="text-align:left"></th>
      <th style="text-align:left"></th>
      <th style="text-align:left"></th>
      <th style="text-align:left"></th>
      <th style="text-align:left"></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:left"></td>
      <td style="text-align:left"></td>
      <td style="text-align:left"></td>
      <td style="text-align:left"></td>
      <td style="text-align:left"></td>
      <td style="text-align:left"></td>
    </tr>
    <tr>
      <td style="text-align:left">Simulator</td>
      <td style="text-align:left">Engine</td>
      <td style="text-align:left">LearningSupport</td>
      <td style="text-align:left">OpenSource?</td>
      <td style="text-align:left">Unique Features</td>
      <td style="text-align:left">Simulator</td>
    </tr>
    <tr>
      <td style="text-align:left">Deepdrive</td>
      <td style="text-align:left">Unreal Engine</td>
      <td style="text-align:left">Tensorflow</td>
      <td style="text-align:left">YES</td>
      <td style="text-align:left">+ High FPS due to shared memory+ Terrain is not just flat+ Realistic rendering+
        100GB (8.2 hours) dataset available- Only ground vehicles- Sparse documentation</td>
      <td
      style="text-align:left">Deepdrive</td>
    </tr>
    <tr>
      <td style="text-align:left">Carla</td>
      <td style="text-align:left">Unreal Engine</td>
      <td style="text-align:left">TensorFlowChainer</td>
      <td style="text-align:left">YES</td>
      <td style="text-align:left">+ Easy to get started- Training code currently unavailable</td>
      <td style="text-align:left">Carla</td>
    </tr>
    <tr>
      <td style="text-align:left">Microsoft AirSim</td>
      <td style="text-align:left">Unreal Engine</td>
      <td style="text-align:left">CNTK</td>
      <td style="text-align:left">YES</td>
      <td style="text-align:left">+ UAV support+ Realtime hardware controller support</td>
      <td style="text-align:left">Microsoft AirSim</td>
    </tr>
    <tr>
      <td style="text-align:left">Baidu Apollo</td>
      <td style="text-align:left">ROSDreamview</td>
      <td style="text-align:left">TensorflowKeras</td>
      <td style="text-align:left">YES</td>
      <td style="text-align:left">+ Support includes free Udacity Apollo course- Hardware focused- Only
        tested on Lincoln MKZ</td>
      <td style="text-align:left">Baidu Apollo</td>
    </tr>
    <tr>
      <td style="text-align:left">Ardupilot</td>
      <td style="text-align:left">SITL Simulator Gazebo</td>
      <td style="text-align:left">GymFC</td>
      <td style="text-align:left">YES</td>
      <td style="text-align:left">+ UAV support- Hardware focused- Not initially intended for Learning</td>
      <td
      style="text-align:left">Ardupilot</td>
    </tr>
    <tr>
      <td style="text-align:left">TORCS</td>
      <td style="text-align:left">OpenGL</td>
      <td style="text-align:left">TensorflowKeras</td>
      <td style="text-align:left">YES</td>
      <td style="text-align:left">+ Easy to get started+ Primary research platform for years- Outdated graphics</td>
      <td
      style="text-align:left">TORCS</td>
    </tr>
    <tr>
      <td style="text-align:left">Nvidia DriveWorks</td>
      <td style="text-align:left">Unreal Engine</td>
      <td style="text-align:left">Unknown</td>
      <td style="text-align:left">?</td>
      <td style="text-align:left">+ Good support for Nvidia GPUs and boards+ Used on Tesla Vehicles</td>
      <td
      style="text-align:left">Nvidia DriveWorks</td>
    </tr>
    <tr>
      <td style="text-align:left">Zoox</td>
      <td style="text-align:left">Unreal Engine</td>
      <td style="text-align:left">Unknown</td>
      <td style="text-align:left">NO</td>
      <td style="text-align:left">
        <ul>
          <li>Raised $790,000,000</li>
        </ul>
      </td>
      <td style="text-align:left">Zoox</td>
    </tr>
    <tr>
      <td style="text-align:left">Cruise Automation</td>
      <td style="text-align:left">Unreal Engine</td>
      <td style="text-align:left">Unknown</td>
      <td style="text-align:left">NO</td>
      <td style="text-align:left">
        <ul>
          <li>Raised $3,368,800,000</li>
        </ul>
      </td>
      <td style="text-align:left">Cruise Automation</td>
    </tr>
  </tbody>
</table>

Common notes:

* All simulators require dedicated GPU
* Their usage is not always trivial
* SITL: Software in the Loop
* Since our objective is for testing in a simulation environment, hardware focused simulators have been given a negative weightage

Knowlegde prerequisites:

* Linux
* Python
* GitHub
* Docker \(Apollo only\)
* Machine Learning Practices

