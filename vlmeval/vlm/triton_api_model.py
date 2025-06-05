import re
import json
import base64
import queue
import hashlib
import os
import sys
import time
import argparse
import uuid
import warnings
from threading import Thread
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient 
from tritonclient.utils import InferenceServerException
from PIL import Image
from .base import BaseModel

def construct_messages(prompt, img_base64_list):
    """
    根据传入的 prompt 和图片列表构造消息。
    prompt 中已包含正确的 <image> 或 <image数字> 占位符，
    消息的顺序将依照占位符与文本分割后构造：
      - 占位符位置构造图片消息，
      - 文本片段构造文本消息。
    """
    #这里只需要修改<image>为<image/d*>就可以拥有多图能力
    parts = re.split(r'(<image>)', prompt)
    messages = []
    img_idx = 0

    for part in parts:
        if re.fullmatch(r'<image>', part):
            if img_idx >= len(img_base64_list):
                raise ValueError("图片占位符数量超过实际图片数")
            messages.append({
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": img_base64_list[img_idx]}
                }]
            })
            img_idx += 1
        else:
            text = part.strip()
            if text:
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": text
                    }]
                })
    if img_idx < len(img_base64_list):
        for remaining in img_base64_list[img_idx:]:
            messages.append({
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": remaining}
                }]
            })
    return {"messages": messages}

def string_to_hash(text):
    hash_object = hashlib.sha256()
    byte_text = text.encode('utf-8')
    hash_object.update(byte_text)
    return hash_object.hexdigest()

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)

def inference(server_url, verbose, stream_timeout, prompt, img_list, model_name='bls', custom_request=None):
    sequence_id = 100
    result_list = []
    stop_flag = False
    user_data = UserData()

    def show_results(q):
        nonlocal stop_flag
        while not stop_flag:
            try:
                data_item = user_data._completed_requests.get(timeout=0.001)
            except queue.Empty:
                continue
            if isinstance(data_item, InferenceServerException):
                print('InferenceServerException:', data_item)
                sys.exit(1)
            result = data_item.as_numpy("OUTPUT")
            result_list.append(result)
        q.put(''.join([item[0][0].decode('utf-8') for item in result_list]))

    q = queue.Queue()
    res_thd = Thread(target=show_results, args=(q,))
    res_thd.start()

    with grpcclient.InferenceServerClient(url=server_url, verbose=verbose) as triton_client:
        try:
            triton_client.start_stream(
                callback=partial(callback, user_data),
                stream_timeout=stream_timeout,
            )

            inputs = []
            if custom_request:
                request_str = json.dumps(custom_request)
            else:
                messages_dict = construct_messages(prompt, img_list)
                request = {
                    **messages_dict,
                    "max_new_tokens": 2048,
                    "temperature": 0.95,
                    "top_p": 0.7,
                    "top_k": 4,
                    "repetition_penalty": 1.1
                }
                # print("==== inference 内部构造的 messages ====")
                # print(json.dumps(request["messages"], indent=2, ensure_ascii=False))
                # print("=================================")
                request_str = json.dumps(request)

            request_bytes = np.array([request_str.encode("utf-8")], dtype=bytes).reshape([1, -1])
            inputs.append(grpcclient.InferInput('REQUEST', request_bytes.shape, "BYTES"))
            inputs[0].set_data_from_numpy(request_bytes)

            outputs = [grpcclient.InferRequestedOutput("OUTPUT")]

            triton_client.async_stream_infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs,
                request_id=str(sequence_id),
                sequence_id=sequence_id,
                sequence_start=True,
                sequence_end=True,
            )

            triton_client.stop_stream()
            stop_flag = True
            res_thd.join()

        except InferenceServerException as error:
            print(error)
            sys.exit(1)

    return q.get()

class TritonAPIModel(BaseModel):
    def __init__(self, **kwargs):
        self.model_name = 'LongCat-VL-Medium-ETFA-vlm_pipeline'
        self.stop_flag = False
        self.server_url = '10.164.245.74:26384'
        print(f'TritonAPIModel init: model_name: {self.model_name}, triton server url: {self.server_url}')

    def message_to_promptimg(self, message, dataset=None):
        """
        直接解析 message，文本部分聚合为 prompt，图片部分保持原有顺序。
        """
        # print(f"message_to_promptimg message: {message}\n\n")

        texts = [x['value'] for x in message if x['type'] == 'text']
        images = [x['value'] for x in message if x['type'] == 'image']
        prompt = '\n'.join(texts)
        return prompt, images if images else None

    def generate_inner(self, message, dataset=None):
        prompt, images = self.message_to_promptimg(message, dataset=dataset)
        prompt = prompt.replace('<image>', '')
        # print(f"prompt: {prompt},images: {images}")


        print(f'TritonAPIModel: generate_inner images: {images}\n, prompt: {prompt}\n') 

        img_base64_list = [
            f"data:image/png;base64,{base64.b64encode(open(img, 'rb').read()).decode('utf-8')}"
            for img in images
        ]
        # base_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/zhoujinhao03/project/data_root/data_root/"
        # url_prefix = "http://10.166.132.61:33333/"

        # img_base64_list = [
        #     f"{url_prefix}{img[len(base_path):]}"
        #     for img in images
        # ]
        result_str = inference(
            stream_timeout=None,
            server_url=self.server_url,
            verbose=False,
            prompt=prompt,
            img_list=img_base64_list,
            model_name=self.model_name,
            custom_request=None
        )
        return result_str
