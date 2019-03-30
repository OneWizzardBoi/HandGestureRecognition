# HandGestureRecognition

HandGestureRecognition is a project I made as a member of the [SynapETS](https://synapsets.etsmtl.ca/) student club at [ETS](https://www.etsmtl.ca/en) in Montreal. The SynapETS student club focuses on developing brain/machine interfaces and exploring biomedical technologies.

The main goal of this project is to experiment with different signal processing and classification techniques to interpret biological signals. Here, the focus is on interpreting the bio-electrical signals from the forearm in order to recognize different movement patterns. This kind of tech can be used for ergonomic applications (ex: [CTRL-Labs](https://www.ctrl-labs.com/)) or biomedical applications (ex: [Open Bionics](https://openbionics.com/)).

![](Docs/demo_gif.gif)

I made the project as modular as possible by dividing it into 3 software components. All components are separately documented and come with installation steps and detailed usage instructions.

**Note that this is meant to be an academic/research project and not an optimal solution for a scalable movement recognition.**

The SynapsETS team plans to keep working on the project and add new components in the future. You should check out their branch [here](https://google.ca).

### Components
* [emg_comm](emg_comm/)
* [pose_recognition](pose_recognition/)
* [pose_visualizer](pose_visualizer/)

### License
This project is licensed under the MIT License - see the LICENSE.md file for details.