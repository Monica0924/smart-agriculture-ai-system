# Smart Agriculture AI System

A complete AI-powered farm monitoring and prediction system for cows, chickens, and crops.
This system uses image classification, audio analysis, time-series prediction, and profit calculation to help farmers make better decisions.

# Features
**🐄 Cow Module**

Predict daily milk yield (litres/day).

Analyze cow audio to detect whether the cow is healthy or stressed.

Identify cow breed using an uploaded image.

Calculate milk profit based on current date, time, and market rate.

**🐔 Chicken Module**

Predict weekly egg count from a chicken.

Identify chicken breed using an uploaded image.

Calculate egg profit using current market prices.

**🌾 Crop Module**

Predict crop profit based on yield, time, and market prices.

Identify crop type from an image.

Detect crop health and diseases using image analysis.

# AI Models Used
Task                                	Suggested Algorithm

Milk yield prediction	               Random Forest Regression

Egg count prediction	               Time-series LSTM

Animal Breed Detection	             CNN / MobileNetV3

Crop Type Detection                  CNN / EfficientNet

Audio-based Health Detection	       Audio Spectrogram + CNN

Crop Disease Detection             	 CNN (PlantVillage dataset baseline)
