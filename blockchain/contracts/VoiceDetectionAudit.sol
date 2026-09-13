// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract VoiceDetectionAudit {

    struct AuditRecord {
        bytes32 audioHash;
        string prediction;
        uint256 riskScore;
        string riskLevel;
        uint256 timestamp;
        address recorder;
    }

    uint256 public auditCount;

    mapping(uint256 => AuditRecord) public audits;

    event AuditRecorded(
        uint256 indexed auditId,
        bytes32 indexed audioHash,
        string prediction,
        uint256 riskScore,
        string riskLevel,
        uint256 timestamp,
        address recorder
    );

    function recordAudit(
        bytes32 _audioHash,
        string memory _prediction,
        uint256 _riskScore,
        string memory _riskLevel
    ) public {

        require(_riskScore <= 100, "Risk score must be 0-100");

        auditCount++;

        audits[auditCount] = AuditRecord(
            _audioHash,
            _prediction,
            _riskScore,
            _riskLevel,
            block.timestamp,
            msg.sender
        );

        emit AuditRecorded(
            auditCount,
            _audioHash,
            _prediction,
            _riskScore,
            _riskLevel,
            block.timestamp,
            msg.sender
        );
    }

    function getAudit(uint256 _auditId)
        public
        view
        returns (
            bytes32 audioHash,
            string memory prediction,
            uint256 riskScore,
            string memory riskLevel,
            uint256 timestamp,
            address recorder
        )
    {
        require(_auditId > 0 && _auditId <= auditCount, "Audit does not exist");

        AuditRecord memory record = audits[_auditId];

        return (
            record.audioHash,
            record.prediction,
            record.riskScore,
            record.riskLevel,
            record.timestamp,
            record.recorder
        );
    }
}