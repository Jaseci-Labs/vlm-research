from openai import OpenAI
import base64
import re
import json

quota_end = False

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)
def response(image_url):
  completion = client.chat.completions.create(
    model="qwen/qwen2.5-vl-32b-instruct:free",
    messages=[
      {
            "role": "system",
            "content": "You are an expert in analyzing car damage and writing insurance reports based on provided images and descriptions."
          },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Please provide response based on the provided output format. Expected output format:\n```json\n{\n  \"predictions\": [\n    {\n      \"location\": \"front bumper\",\n      \"damage_type\": \"dent\",\n      \"severity\": \"major\"\n    },\n    {\n      \"location\": \"driver side door\",\n      \"damage_type\": \"scratch\",\n      \"severity\": \"minor\"\n    }\n  ],\n \"report\":\"Insurance Report: The vehicle sustained significant damage, including a major dent on the front bumper and a minor scratch on the driver side door. Estimated repair cost: $1,500.\"}\n```"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": image_url
            }
          }
        ]
      }
    ]
  )
  if completion.choices is None:
     print("Quota end")
     return None
  out = completion.choices[0].message.content
  return out

with open("/home/thami/prog/imagelinks.txt", "r") as f:
    lines = f.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].strip()

res_file = {}

for i in range(10,12):
    match = False
    n = 0
    while not match and n < 5 and not quota_end:
      n += 1
      res = response(lines[i])
      if res is None:
        quota_end = True
        break
      match = re.search(r'{.*}', res, re.DOTALL)
      if not match and n==5:
        with open("/home/thami/prog/problem.txt", "a") as f:
          f.write(lines[i] + "\n")
    if not match and not quota_end:
        print("No JSON found")
        continue
    if quota_end:
        print("Quota end")
        break
    json_str = match.group(0)
    data = json.loads(json_str)
    print(data)
    print(lines[i], "image is done")
    res_file[lines[i]] = data

with open("/home/thami/prog/res.json", "r") as f:
    try:
        old_res_file = json.load(f)
    except json.JSONDecodeError:
        old_res_file = {}

old_res_file.update(res_file)
with open("/home/thami/prog/res.json", "w") as f:
    json.dump(old_res_file, f, indent=4)
print("completely done")