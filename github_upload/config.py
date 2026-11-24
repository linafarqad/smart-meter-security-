

N_METERS = 500           
GROUP_SIZE = 20           
N_DAYS = 365              
SLOTS = 48               
T = N_DAYS * SLOTS       
WARMUP_DAYS = 56         
N_SEEDS = 30              

MEAN_DAILY_KWH = 24.0     
LOGNORM_SIGMA = 0.35      
WEEKEND_FACTOR = 1.10     
SEASONAL_AMPLITUDE = 0.20 
NOISE_SD = 0.10           
RESOLUTION_KWH = 0.001    


P_PACKET_LOSS = 0.003           
METER_OUTAGE_PER_DAY = 1 / 60  
METER_OUTAGE_LEN = (1, 6)       
GROUP_OUTAGE_PER_DAY = 1 / 90   
GROUP_OUTAGE_LEN = (2, 8)       
VACATION_PER_YEAR = 0.5         
VACATION_DAYS = (3, 10)
VACATION_FACTOR = (0.25, 0.50) 
WEATHER_EVENTS = 6              
WEATHER_DAYS = (2, 5)
WEATHER_FACTOR = (1.20, 1.40)
LEGIT_DISCONNECT_PER_YEAR = 0.2 
LEGIT_DISCONNECT_DAYS = (1, 3)


CAMPAIGNS_PER_RUN = 8
S1_ALPHA = (0.5, 0.9)
S1_BETA = (-0.05, 0.0)          
S1_FRACTION = (0.01, 0.20)
S1_DAYS = (1, 7)
S2_HOURS = (2, 24)
S2_SINGLE_METERS = (1, 5)
S2_GROUP_FRACTION = (0.10, 0.30)
S2_GROUPS = (1, 3)
S3_HOURS = (6, 48)
S3_FRACTION = (0.01, 0.20)
S3_REPLAY_LAG = 7 * SLOTS       
S4_FRACTION = (0.05, 0.30)
S4_OFF_INTERVALS = (4, 24)      


D1_WINDOW_DAYS = 28
D1_Z = 3.0
D1_K = 3
D2_GAP = 4                     
D2_REPEAT = 4                 
D2_MAX_KWH = 7.5                
D2_BURST_FRACTION = 0.01       
D3_Z = 3.5
D3_K = 3
D3_MIN_SILENT = 2               
D3_STATUS_MIN = 3               
