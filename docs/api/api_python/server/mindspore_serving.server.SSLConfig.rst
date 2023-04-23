﻿
.. py:class:: mindspore_serving.server.SSLConfig(certificate, private_key, custom_ca=None, verify_client=False)

    Serving服务器中，使能gRPC或RESTful服务器SSL功能时，SSL的参数配置。

    参数：
        - **certificate** (str) - PEM编码的证书链内容，如果值为 ``None``，则表示不使用证书链。
        - **private_key** (str) - PEM编码的私钥内容，如果值为 ``None``，则表示不使用私钥。
        - **custom_ca** (str, 可选) - PEM编码的根证书内容。当 `verify_client` 为 ``True`` 时， `custom_ca` 必须指定。当 `verify_client` 为 ``False`` 时，将忽略此参数。默认值：``None``。
        - **verify_client** (bool, 可选) - 如果 `verify_client` 为 ``True``，则启用客户端服务器双向认证。如果为 ``False``，则仅启用客户端对服务器的单向认证。默认值：``False``。

    异常：
        - **RuntimeError** - 参数的类型或值无效。
