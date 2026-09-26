# AxisAllies
Neural network model AI for strategy board game.
An MCTS self-play data collection system with full-fledged training pipeline. This project is intended to demonstrate how a network model can be trained and used
for strategy game AI. The plugin in the project is not intended to be ripped and used in external projects.

For that purpose, a plugin may be created where users may use this turnkey system of training and using a network model in their game. The major caveat
is defining the legal action spaces (policy head sizes), game state, legal action masking, and phase logic. The methods in this project are hard-wired
to the overlying game. A plugin dedicated for streamlined use in other projects may be in development, but is not the purpose here.

The Unreal Engine project uses version 5.6 and is built in Visual Studio. There is a build available in the Windows folder. As training improves,
the model's .onnx and .pt files will be updated there for both the build and the Unreal Engine project's content folder.
