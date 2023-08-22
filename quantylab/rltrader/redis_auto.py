import redis
import json



class redis_self:
    

    def pub_msg(channel, vale):
        r = redis.StrictRedis(host='34.64.240.96', port=6379, db=0)    
        r.publish(channel, json.dumps(vale)) # vale 키값
    

    def sub_msg(channel):
       r = redis.StrictRedis(host='34.64.240.96', port=6379, db=0)
       s = r.pubsub()
       s.subscribe(channel)
       for message in s.listen():
            # print(type(message))
            if message['type'] == 'message':
                  res_data = message['data']
                  if isinstance(res_data, bytes):
                     res_data = res_data.decode()
                     try:
                        res_dict = json.loads(res_data)
                        return res_dict
                     except json.JSONDecodeError:
                        print(res_data)
                  break
       return None
