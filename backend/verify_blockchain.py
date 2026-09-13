import json
from web3 import Web3

RPC_URL = "http://127.0.0.1:8545"

CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"

ABI_PATH = r"C:\Users\hp\Desktop\AI-Voice-Clone-Detection\blockchain\artifacts\contracts\VoiceDetectionAudit.sol\VoiceDetectionAudit.json"


# Connect to Hardhat blockchain
w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    print("❌ Blockchain not connected")
    exit()

print("✅ Blockchain connected")


# Load ABI
with open(ABI_PATH, "r") as f:
    contract_data = json.load(f)

contract = w3.eth.contract(
    address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=contract_data["abi"]
)


# Get total audits
audit_count = contract.functions.auditCount().call()

print("\n================================")
print("BLOCKCHAIN AUDIT VERIFICATION")
print("================================")

print(f"Total audits: {audit_count}")


if audit_count == 0:
    print("❌ No audit records found")
    exit()


# Read Audit ID 1
audit = contract.functions.getAudit(1).call()

audio_hash = audit[0].hex()
prediction = audit[1]
risk_score = audit[2]
risk_level = audit[3]
timestamp = audit[4]
recorder = audit[5]


print("\nAudit ID       :", 1)
print("Audio Hash     :", audio_hash)
print("Prediction     :", prediction)
print("Risk Score     :", risk_score)
print("Risk Level     :", risk_level)
print("Timestamp      :", timestamp)
print("Recorder       :", recorder)

print("\n================================")
print("✅ BLOCKCHAIN RECORD VERIFIED")
print("================================")
