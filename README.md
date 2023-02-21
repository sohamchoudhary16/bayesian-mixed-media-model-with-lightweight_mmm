# bayesian-mixed-media-model-with-lightweight_mmm

## Situation
There is a business X that operates an online store. X pays weekly fees to advertise on seven different paid channels. Ads and campaigns in a week typically influence sales in the weeks that follow, with marketing initiatives typically not having an instant impact. As a result, it is obviously very intriguing to see how beneficial various channels are for a corporation. Think of TV, radio, billboards, and internet advertisements like Google Ads, Facebook Ads, etc. while imagining channels. So, it is possible to anticipate that different channels would target different consumers at different periods and, as a result, have quite diverse effects on future sales. 

## Dataset dataset.csv
* start_of_week: indicates the first day of the week
* revenue: indicated the revenue that was generated in current week from sales
* spend_channel_1 to spend_channel_7: indicates the cost of marketing spend in current week respective to the channel

## Solution MMM_notebook.ipynb
The situation has been dealt with the use of lightweight_mmm (https://github.com/google/lightweight_mmm). The LightweightMMM package (built using Numpyro and JAX) helps advertisers easily build Bayesian MMM models by providing the functionality to appropriately scale data, evaluate models, optimise budget allocations and plot common graphs used in the field. The notebook provides answers to topics of model spend carry over, prior sampling vs. posterior sampling, main insights in terms of channel performance/ effects and ROI (return on investment) estimates per channel
