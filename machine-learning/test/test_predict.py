import httpx
import asyncio

async def send_request():
    url = 'http://localhost:3003/predict'
    data = {
        'modelName': 'buffalo_l',
        'modelType': 'facial-recognition',  # Replace with actual enum value if necessary
        'options': '{"minScore":0.034}',  # JSON string of options
    }

    # Open the image file in binary mode
    # image_path= '/workspaces/frigate/videos/faces/Tuan_Anh/Tuan_Anh_0003.jpg'
    image_path = './person.jpg'
    with open(image_path, 'rb') as image_file:
        files = {'image': ('Tuan_Anh_0003.jpg', image_file, 'image/jpeg')}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data, files=files)
            print(f"Status Code: {response.status_code}")
            print("Response JSON:", response.json())
            print("type: ", type(response.json()))
            print(response.json()[0]['boundingBox'])
# Run the async function
asyncio.run(send_request())
