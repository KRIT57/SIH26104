import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

const VoiceDetectionAuditModule = buildModule("VoiceDetectionAuditModule", (m) => {
  const audit = m.contract("VoiceDetectionAudit");

  return { audit };
});

export default VoiceDetectionAuditModule;