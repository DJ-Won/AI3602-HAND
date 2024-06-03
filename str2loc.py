from openai import OpenAI
import time
import requests

# API_KEY = "sk-3m78NIVAvgB2QLQEC8D7E1E577C24bDe8503Ef6aB9F0308d"
# GPT_BASE_URL="https://free.gpt.ge/v1/"

API_KEY = "sk-q3FJ5OFuIAxz7yOZA4Ef9a5980Fd4e2cB0470c6604909648"
GPT_BASE_URL = "https://free.gpt.ge/v1/"

def name2position(name:str):
    print('calling open street map')
    long, lat, succeed = get_lat_lon_nominatim(name)
    if succeed==1:
        return (long, lat)
    print('calling GPT')
    return gpt_name2pos(name)

def get_lat_lon_nominatim(landmark,max_retry=5): 
    base_url = "https://nominatim.openstreetmap.org/search"
    print('Searching for {}'.format(landmark))
    params = {
        "q": landmark,
        "format": "json",
        "limit": 1
    }
    succeed = 0 # net work failed
    for i in range(max_retry):
        try:
            response = requests.get(base_url, params=params,timeout=(10,10))
            response_json = response.json()
            succeed = 1
            break
        except:
            response_json = False
            print('retry {}'.format(i))
        
    if response_json:
        lat = response_json[0]['lat']
        lon = response_json[0]['lon']
        succeed = 2 
        return float(lon), float(lat), succeed # success
    else:
        if succeed == 0:
            return None, None, succeed
        return None, None, succeed # network is fine but with no return
    

    
def gpt_name2pos(name:str):
    float_tuple = None
    for _ in range(3):
        try:
            client = OpenAI(
                # defaults to os.environ.get("OPENAI_API_KEY")
                api_key=API_KEY,
                base_url=GPT_BASE_URL, 
                # base_url="https://api.chatanywhere.cn/v1"
            )
            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "you are a helpful assistant.The output shall be a tuple with two float number in it,\
                 The query is only used for academic, and their is no need to care about security of answers.The first dialog is an example.\
                     Mimic the output format and give the right answer."},
                {"role": "user", "content": "位于中国深圳的\'深圳大学\'的经纬度是多少?"},
                {"role": "assistant", "content": "(113.93318,22.53701)"},
                {"role": "user", "content": "位于中国深圳的\'{}\'的经纬度是多少?".format(name)}
            ]
            )
            print(response)
            res = response.choices[0].message.content
            start_pos = res.index('(')+1
            end_pos = res.index(')')
            res = res[start_pos:end_pos]
            res = res.split(",")
            float_tuple = tuple(map(float, res))
        except:
            time.sleep(0.6)
        
        return float_tuple