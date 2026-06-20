def statsapi(API_KEY):
    import http.client

    conn = http.client.HTTPSConnection("v3.football.api-sports.io")

    headers = {
        'x-apisports-key': API_KEY,
        'id' : 6
        }


    conn.request("GET", "/leagues/", headers=headers)

    res = conn.getresponse()
    data = res.read()
    print('OUTPUT')
    #print(data.decode("utf-8"))
    type(data)