import requests
import ssl
import json
import csv
import time

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

flag = 0

for blocknum in range(50):
    response = requests.get('https://sochain.com/api/v2/get_block/BTC/'+str(blocknum+500000))
    # uh = urllib.request.urlopen(address, context=ctx)

    while response.status_code != 200:
        response = requests.get('https://sochain.com/api/v2/get_block/BTC/'+str(blocknum+500000))

    #data = uh.read()
    #print('Retrieved', len(data), 'characters')
    #print(data.decode())
    info = response.json()
    if flag == 1:
        print(info['data']['block_no'], info['data']['blockhash'], len(info['data']['txs']))
        print(info['data']['merkleroot'])

    for txid in info['data']['txs']:
        if flag == 0:
            if txid == "caa038226dad54ebd170c0cb60b342a371b3a0109c4bf12094be8db2eae5cffd":
                flag = 1
            else:
                continue

        print(txid)        
        txaddr = requests.get('https://sochain.com/api/v2/get_tx/BTC/'+ txid)
        
        while txaddr.status_code != 200:
            txaddr = requests.get('https://sochain.com/api/v2/get_tx/BTC/'+ txid)

        txinfo = txaddr.json()
        # print(txinfo)

        print(len(txinfo['data']['inputs']))
        for txin in txinfo['data']['inputs']:
            #print(txin)
            if type(txin['from_output']) is dict:
                print(txin['from_output']['txid'], txin['from_output']['output_no'])
            else:
                print("0")

        print(len(txinfo['data']['outputs']))
