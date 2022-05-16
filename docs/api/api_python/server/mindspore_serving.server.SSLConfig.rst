
.. py:class:: SSLConfig(certificate, private_key, custom_ca=None, verify_client=False)

    Serving服务器中，gRPC或RESTful服务器SSL的参数配置。

    **参数：**

    - **certificate** (str) - PEM编码的证书链内容，如果值为None，则表示不使用证书链。
    - **private_key** (str) - PEM编码的私钥内容，如果值为None，则表示不使用私钥。
    - **custom_ca** (str, optional) - PEM编码的根证书内容。当 `verify_client` 为True时， `custom_ca` 必须指定。当 `verify_client` 为False时，将忽略此参数。默认值：None。
    - **verify_client** (bool, optional) - 如果 `verify_client` 为True，则启用客户端服务器双向认证。如果为False，则仅启用客服端对服务器的单向认证。默认值：False。

    **异常：**

    - **RuntimeError** - 参数的类型或值无效。
