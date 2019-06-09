pragma solidity 0.4.21;

import "blockchain/Query.sol";


contract Delegator {
    address public owner;

    mapping(address => uint256[]) public queries;
    mapping(uint256 => address) private addressBook;

    address public q;
    address[] public addrList;

    event QueryCreated(address,address);

    function Delegator() public {
        owner = msg.sender;
    }

    function query(address target)
        public
    {
        int[] memory lst = new int[](3);
        lst[0] = 0;
        lst[1] = 1;
        lst[2] = 2;
        addrList.push(target);
        q = address(new Query(1, 1, addrList));
        pingClients(addrList);
    }

    function pingClients(address[] clientList) {
        uint clientLen = clientList.length;
        for (uint i = 0; i < clientLen; i++) {
            emit QueryCreated(clientList[i], q);
        }
    }
}
