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
    



    def pub_setlist(key,vale):
       r = redis.StrictRedis(host='34.64.240.96', port=6379, db=0)
       r.set(key,json.dumps(vale))

    def sub_getlist(key):
       r = redis.StrictRedis(host='34.64.240.96', port=6379, db=0)
       c = r.get(key)
       cb = json.loads(c)
       return cb



    def pub_newmsg(channel,value):
      redis_host = '34.64.240.96'
      redis_port = 6379
      r = redis.StrictRedis(host=redis_host, port=redis_port, db=0)
      json_data = json.dumps(value)
      prev_value = r.get('h')
      if prev_value:
         prev_json_data = json.loads(prev_value)
         if json_data == prev_json_data:
               # If the data is the same as the previous value, return without publishing
               print("Data is the same as the previous value. Not publishing.")
               return
      # Publish the new data
      r.publish(channel, json_data)
      # Store the current value in Redis
      r.set('h', json_data)