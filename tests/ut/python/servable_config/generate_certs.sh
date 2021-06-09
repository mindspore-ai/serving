#!/bin/bash
echo "[req]
default_bits = 2048
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no
[req_distinguished_name]
countryName = XX
stateOrProvinceName = Self-signed Cert
commonName = Self-signed Cert
[v3_req]
basicConstraints = CA:TRUE" > ca.cnf

# generate ca's cert and private key for signing server and client cert
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ca.key -out ca.crt -config ca.cnf

rm ca.cnf

# generate server's cert

IP=$SERVING_IP
DNS=$SERVING_HOSTNAME
CN=$SERVING_COMMON_NAME

echo "
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names
[alt_names]
IP.1 = $IP
DNS.1 = $DNS
" > server.cnf

openssl genrsa -out server.key 2048

openssl req -new -key server.key -out server.csr -subj "/C=XX/ST=MyST/L=XX/O=HW/OU=gRPC/CN=$CN"

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 730 -sha256 -extfile server.cnf

rm server.cnf

# generate client's cert

openssl genrsa -out client.key 2048

openssl req -new -key client.key -out client.csr -subj "/C=XX/ST=MyST/L=XX/O=HW/OU=gRPC/CN=client"

openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt -days 730 -sha256