pragma solidity 0.4.21;


contract Query {

    uint public vectorLength;
    uint public totalNumData = 0;
    uint public numberOfResponses = 0;
    uint public numberDone = 0;
    uint public maxRounds;
    uint public numClients;
    uint public currentRound;
    uint n_k;

    string public ipfsaddr = 'QmVm4yB2jxPwXXVXM6n86TuwA4jCQ7EfNPjguFrhoCbPiJ';
    
    int[] currentWeights;
    address[] addrList;
    
    mapping(int => int[]) public weights;

    /////////////
    // Structs //
    /////////////

    ///////////////
    // Modifiers //
    ///////////////

    //////////////
    //  Events  //
    //////////////

    event ClientSelected(address,string);
    event ResponseReceived(uint256);
    event BeginAveraging(string);
    // event FederatedAveragingComplete();

    ///////////////
    // Functions //
    ///////////////

    function Query(int _maxRounds, int _numClients, address[] _addrList) public {

        maxRounds = uint(_maxRounds);
        numClients = uint(_numClients);

        currentRound = 1;

        addrList = _addrList;
    }

    function pingClients() {
        uint addrLen = addrList.length;
        for (uint i = 0; i < addrLen; i++) {
            emit ClientSelected(addrList[i], ipfsaddr);
        }
        // currentWeights = new int[](vectorLength);
    }

    function receiveResponse(
        string IPFSaddress,
        uint _n_k) 
        public
        {
            numberOfResponses++;
            n_k = _n_k;
            if (numberOfResponses > 1) {
                emit BeginAveraging(IPFSaddress);
            } else {
                numberDone++;
            }
            //TODO: make sure that only a client who is SUPPOSED to be training can call this!
            //received client update has already been multiplied by _numData=n_k
            //when we send it back we expect the client to divide it by /Sigma(_numData)=n
            // uint numData = uint(_numData);
            // totalNumData = totalNumData + _numData;
            // uint i;
            // // int[] memory newUpdate = new int[](vectorLength);
            // //below is commented out because we don't need to divide yet
            // // for (i = 0; i < vectorLength; i ++) {
            // //     newUpdate[i] =  divide(_clientUpdate[i], totalNumData);
            // // }
            // //adding the new client update to the current ones 
            // for (i = 0; i < vectorLength; i++) {
            //     currentWeights[i] = currentWeights[i] + _clientUpdate[i];
            // }
            // numberOfResponses++;
            // //if this was the last client we needed to hear from, go ahead and 
            // //start another round of training if we haven't exceeded our max rounds
            // if (numberOfResponses > numClients) {
            //     //scale our weights down
            //     for (i = 0; i < vectorLength; i++) {
            //     currentWeights[i] = divide(currentWeights[i], totalNumData);
            // }
            //     if (currentRound < maxRounds) {
            //         // pingClients(keyList);
            //     } else {
            //         //we're done!
            //     }
            //     currentRound++;
            // }
            //now transfer some ETH to the client as thanks for training our model!
            //for now the heuristic is just how much data they had
            //TODO: get the address from the client that sent in their response
            // _clientAddress.transfer(numData);
        }
    function allDone(string IPFSaddress) public {
        ipfsaddr = IPFSaddress;
        numberDone++;
    }
    function divide(int i, uint j) internal pure returns (int) {
        //TODO: Implement real division lmao
        return i / int (j);
    }
    function calculateAccuracy() internal pure returns (uint) {

    }
    function terminate() internal {
        
    }
}
    // function sendResponse(
    //     int[] update,
    //     int key,
    //     int numData)
    //     external
    //     returns (int[])
    //     // needs permissioning
    // {
    //     uint i;
    //     uint keyLen = keyList.length;
    //     if (moreThanOne) {
    //         int[] memory newUpdate = new int[](vectorLength);
            
    //         // scaling
    //         for (i = 0; i < keyLen; i++) {
    //             newUpdate[i] = update[i] * numData;
    //         }

    //         // summation
    //         for (i = 0; i < keyLen; i++) {
    //             currentWeights[i] = currentWeights[i] + newUpdate[i];
    //         }
    //     } else {
    //         for (i = 0; i < keyLen; i++) {
    //             currentWeights[i] = update[i];
    //         }
    //         if (currentWeights.length == vectorLength) {
    //             moreThanOne = true;
    //         }
    //     }
    //     numberOfResponses++;
    //     emit ResponseReceived(numberOfResponses);
    //     emit EventEmit();
    //     return currentWeights;
    // }

    // function inverseScale()
    //     external returns (bool)
    //     // check against threshold
    // {
    //     uint i;
    //     uint j;
    //     uint keyLen = keyList.length;
    //     for (i = 0; i < keyLen; i++) {
    //         for (j = 0; j < vectorLength; j++) {
    //             weights[keyList[i]][j] = weights[keyList[i]][j] / int(totalNumData);
    //         }
    //     }
    //     return true;
    // }
// }
