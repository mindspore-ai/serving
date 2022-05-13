
.. py:class:: mindspore_serving.client.SSLConfig(certificate=None, private_key=None, custom_ca=None)

    Serving服务器gRPC SSL使能时，通过SSLConfig封装SSL证书等相关参数。

    **参数：**

    - **certificate** (str, optional) - PEM编码的证书链内容，如果为None，表示不使用证书链。默认值：None。
    - **private_key** (str, optional) - PEM编码的私钥内容，如果为None，表示不使用私钥。默认值：None。
    - **custom_ca** (str, optional) - PEM编码的根证书内容，如果为None，gRPC运行时将从默认位置加载根证书。默认值：None。

    **异常：**

    - **RuntimeError** - 参数的类型或值无效。
