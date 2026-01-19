This module forecasts electricity price for a selected zone using Temporal Fusion Transformer (TFT) model, using weather data.

Usage:
    - Put token for ENTSO-E and ERA5 in `.env`. 


To run the model

```
python -m cli --zone NL --train --predict
```