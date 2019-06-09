import logging
import shutil
import time

import asyncio
import numpy as np
import tensorflow as tf

from models.perceptron import Perceptron
from models.cnn import CNN
from models.lstm import LSTM

from eth_utils import is_address
from solc import compile_source, compile_files
from blockchain.blockchain_utils import *
from blockchain.ipfs_utils import *
from models.gan import ConversationalNetwork

class Client(object):
    def __init__(self, iden, provider, clientAddress=None, delegatorAddress=None):

        self.web3 = provider
        self.api = ipfsapi.connect('127.0.0.1', 5001)

        self.PASSPHRASE = 'panda'
        self.TEST_ACCOUNT = '0xb4734dCc08241B46C0D7d22D163d065e8581503e'
        self.TEST_KEY = '146396092a127e4cf6ff3872be35d49228c7dc297cf34da5a0808f29cf307da1'

        contract_source_path_A = "blockchain/Delegator.sol"
        contract_source_path_B = "blockchain/Query.sol"
        contract_source_path_C = "blockchain/DAgoraToken.sol"
        self.compiled_sol = compile_files([contract_source_path_A, contract_source_path_B,
        contract_source_path_C])

        self.iden = iden

        if clientAddress:
            assert(is_address(clientAddress))
            self.clientAddress = clientAddress
        else:
            #TODO: Initialize client 'container' address if it wasn't assigned one
            self.clientAddress = self.web3.personal.newAccount(self.PASSPHRASE)
            assert(is_address(self.clientAddress))
        self.web3.personal.unlockAccount(self.clientAddress, self.PASSPHRASE)
    
        print("Client Address:", self.clientAddress)

        self.Delegator_address = delegatorAddress
        self.DAgoraToken_address = self.web3.toChecksumAddress('0x1698215a2bea4935ba9e0f5b48347e83450a6774')

    def get_money(self):
        # get_testnet_eth(self, self.clientAddress, self.web3)
        print("Client balance:", self.web3.eth.getBalance(self.clientAddress))

        Query_id, self.Query_interface = self.compiled_sol.popitem()
        Delegator_id, self.Delegator_interface = self.compiled_sol.popitem()
        self.compiled_sol.popitem()
        self.compiled_sol.popitem()
        DAgoraToken_id, self.DAgoraToken_interface = self.compiled_sol.popitem()

        if self.Delegator_address:
            assert(is_address(self.Delegator_address))
        else:

            # self.Query_address = deploy_Query(self.web3, self.Query_interface, self.TEST_ACCOUNT, addr_lst)
            self.Delegator_address = deploy_Master(self.web3, self.Delegator_interface, self.clientAddress)
            print("Delegator Address", self.Delegator_address)

    def launch_query(self, target_address):
        contract_obj = self.web3.eth.contract(
           address=self.Delegator_address,
           abi=self.Delegator_interface['abi'])
        tx_hash = contract_obj.functions.query(target_address).transact(
            {'from': self.clientAddress})

        self.web3.eth.waitForTransactionReceipt(tx_hash)

        tx_receipt = self.web3.eth.getTransactionReceipt(tx_hash)
        # print(tx_receipt)
        self.Query_address = self.web3.toChecksumAddress('0x' + tx_receipt['logs'][0]['data'].split('000000000000000000000000')[2])
        # return tx_receipt

    def ping_client(self):
        contract_obj = self.web3.eth.contract(
           address=self.Query_address,
           abi=self.Query_interface['abi'])
        tx_hash = contract_obj.functions.pingClients().transact({'from': self.clientAddress})

        self.web3.eth.waitForTransactionReceipt(tx_hash)
        
        tx_receipt = self.web3.eth.getTransactionReceipt(tx_hash)
        return tx_receipt

    def setup_model(self, model_type):
        self.model_type = model_type
        if model_type == "perceptron":
            self.model = Perceptron()
            self.weights_metadata = self.model.get_weights_shape()
        elif model_type == "cnn":
            #TODO: Support CNN
            self.model = CNN()
        elif model_type == "lstm":
            #TODO: Support LSTM
            self.model = LSTM()
        elif model_type == "gan":
            self.model = ConversationalNetwork()
            self.model.build_model(is_retraining=True)
        else:
            raise ValueError("Model {0} not supported.".format(model_type))

    def train(self, IPFSaddress, config):
        #TODO: Make train() only need to take in the config argument ONCE
        logging.info('Training just started.')

        # Get weights from IPFS and load into model
        self.setup_model(config['model_type'])
        ipfs2keras(self.model, IPFSaddress) # fix me

        # Train model
        n_k = self.model.train_model(config)

        # Save new weights to IPFS and return address
        new_model_address = keras2ipfs(self.model)

        return new_model_address, n_k

    def handle_ClientSelected_event(self, event_data):

        e_data = [x for x in event_data.split('00') if x]

        IPFSaddress_receiving = bytearray.fromhex(e_data[3][1:]).decode()[1:]
        # address = self.web3.toChecksumAddress('0x' + e_data[1])
        # assert(self.clientAddress == address)
        print("IPFS address:", IPFSaddress_receiving)

        # IPFS cat from IPFS_receiving

        contract_obj = self.web3.eth.contract(
           address=self.Query_address,
           abi=self.Query_interface['abi'])

        #will be hardcoded
        config = {
            "num_clients": 1,
            "model_type": 'gan',
            "dataset_type": 'iid',
            "fraction": 1.0,
            "max_rounds": 1,
            "batch_size": 10,
            "epochs": 1,
            "learning_rate": 1e-4,
            "save_dir": './results/',
            "goal_accuracy": 1.0,
            "lr_decay": 0.99
        }

        # data = api.cat(IPFSaddress_receiving)

        # IPFS add
        updatedAddress, n_k = self.train(IPFSaddress_receiving, config)

        tx_hash = contract_obj.functions.receiveResponse(updatedAddress, n_k).transact(
            {'from': self.clientAddress})

        self.web3.eth.waitForTransactionReceipt(tx_hash)

        tx_receipt = self.web3.eth.getTransactionReceipt(tx_hash)
        log = contract_obj.events.ResponseReceived().processReceipt(tx_receipt)
        # return log[0]

    def handle_QueryCreated_event(self, event_data):
        print(event_data)
        address = event_data.split("000000000000000000000000")[2]
        assert(is_address(address))
        self.Query_address = self.web3.toChecksumAddress(address)
        return event_data.split("000000000000000000000000")

    def handle_BeginAveraging_event(self, IPFSaddress):
        # get info at address
        contract_obj = self.web3.eth.contract(
           address=self.Query_address,
           abi=self.Query_interface['abi'])
        averagedAddress, num_data = self.runningWeightedAverage(IPFSaddress)
        tx_hash = contract_obj.functions.allDone().transact(
            {'from': self.clientAddress})
        return tx_hash

    def runningWeightedAverage(self, new_model_address, n_k_1, n_k_2):
        weights1, weights2 = self.model.get_weights(), self.preprocess(new_model_address)
        scaled_weights1, scaled_weights2 = model1.scale_weights(weights1, n_k_1), model2.scale_weights(weights2, n_k_2)
        summed_weights = model2.sum_weights(weights1, weights2)
        new_weights = model2.set_weights(summed_weights)
        return keras2ipfs(new_weights), n_k_1 + n_k_2

    def preprocess(self, new_model_address):
        newModel = ConversationalNetwork()
        newModel.build_model(is_retraining=False)
        ipfs2keras(newModel, new_model_address)
        return newModel.get_weights()

    async def start_listening(self, event_to_listen, poll_interval=5):
        while True:
            lst = event_to_listen.get_new_entries()
            if lst:
                # print(lst[0])
                return lst[0]
            await asyncio.sleep(poll_interval)

    def filter_set(self, event_sig, contract_addr, handler):
        event_signature_hash = self.web3.sha3(text=event_sig).hex()
        event_filter = self.web3.eth.filter({
            "address": contract_addr.lower(),
            "topics": [event_signature_hash]
            })
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        try:
            inter = loop.run_until_complete(
                self.start_listening(event_filter, 2))
            check = handler(inter['data'])
        finally:
            loop.close()
        return check

    def main(self):
        check = self.filter_set("QueryCreated(address,address)", self.Delegator_address, self.handle_QueryCreated_event)
        if check[0] + check[1] == self.clientAddress.lower():
            target_contract = check[0] + check[2]
            # print(target_contract)
            # retval = self.filter_set("ClientSelected(address,string)", target_contract, self.handle_ClientSelected_event)
            self.filter_set("ClientSelected(address,string)", target_contract, self.handle_ClientSelected_event)
            # return "I got chosen:", retval[0] + retval[1]
            # print("listening for next round to begin...")
            print("receiving reward...")
            contract_obj = self.web3.eth.contract(
            address=self.DAgoraToken_address,
            abi=self.DAgoraToken_interface['abi'])

            tx_hash = contract_obj.functions.transfer(self.clientAddress, 50000000).transact({'from': '0xf6419f5c5295a70C702aC21aF0f64Be07B59F3c4'})
            self.web3.eth.waitForTransactionReceipt(tx_hash)
            print('sent!')
            # print('Token Balance:', contract_obj.functions.balanceOf(self.clientAddress))
            # alldone = self.filter_set("BeginAveraging(string)", target_contract, self.handle_BeginAveraging_event)
        else:
            return "not me"









    def flatten_weights(self, weights, factor=1e10):
        flattened = []
        for _, tensor in sorted(weights.items()):
            flattened.extend(tensor.flatten().tolist())
        return [int(n*factor) for n in flattened]

    def unflatten_weights(self, flattened, factor=1e10):
        flattened = [n/factor for n in flattened]
        weights = {}
        index = 0
        for name, (shape, size) in sorted(self.weights_metadata.items()):
            weights[name] = np.array(flattened[index:index+size]).reshape(shape)
            index += size
        return weights

    def get_checkpoints_folder(self):
        return "./checkpoints-{0}/{1}/".format(self.iden, self.model_type)

    def get_latest_checkpoint(self):
        return tf.train.latest_checkpoint(self.get_checkpoints_folder())
