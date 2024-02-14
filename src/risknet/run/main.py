from risknet.run import pipeline

first = pipeline.pipeline(fe_enabled=False, baseline=True) #No FE, only feature is credit score
#second = pipeline.pipeline(fe_enabled=False, baseline=False) #No FE, using all original Freddie Mac features
#third = pipeline.pipeline(fe_enabled=True, baseline=False) #Using FE features + original Freddie Mac features