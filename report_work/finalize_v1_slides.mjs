import path from 'node:path';
import { pathToFileURL } from 'node:url';
const SKILL_DIR='C:/Users/LENOVO/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11809/skills/presentations';
const workspaceDir='D:/SIC_Capstone 2026';
const candidatePath=path.join(workspaceDir,'report_work','v1_slides_build','candidate.pptx');
const finalPath=path.join(workspaceDir,'doc','SIC_Capstone_v1_v2_Thuyet_trinh_10_slide.pptx');
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR,'container_tools/artifact_tool_utils.mjs')).href);
const result = await finalizePresentation({
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable:'C:/Users/LENOVO/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
  fontPolicy:{basis:'design',families:['Arial']},
  verifyArtifactToolImport:true,
  receiptPath:path.join(workspaceDir,'report_work','v1_slides_build','validation_v1_v2.json'),
});
console.log(JSON.stringify(result,null,2));
