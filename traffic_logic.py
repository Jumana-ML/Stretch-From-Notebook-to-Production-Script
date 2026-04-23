import logging

# get logger to this file
logger = logging.getLogger(__name__)

def run_traffic_lights():
    lights = ["Red", "Yellow", "Green"]
    
    logger.info("start traffic lights   ")
    
    for i in range(4): # error becuase index out of range
        try:
            current_light = lights[i]
            print(f"traffic light now is : {current_light}")
        except IndexError:
             # here we tell logger that we have error in traffic lights to return error
            logger.error("technical error in traffic lights", exc_info=True)