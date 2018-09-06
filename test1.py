import urllib.request

url = "https://www.douban.com/"
 
request = urllib.request.Request(url)
 
response = urllib.request.urlopen(request)
 
data = response.read()

data = data.decode('utf-8')

print(data)
  
print(type(response))
print(response.geturl())
print(response.info())
print(response.getcode())
