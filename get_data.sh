#!/bin/bash

datadir="NFT_data"

if [ ! -d "$datadir" ]; then
    wget https://github.com/jimmy-academia/dev-NFT/releases/download/nft/NFT_data.zip
    unzip NFT_data.zip
    rm NFT_data.zip
