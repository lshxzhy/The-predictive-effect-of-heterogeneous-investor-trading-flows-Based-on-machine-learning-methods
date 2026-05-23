import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = process.cwd();
const dataPath = path.join(projectRoot, "outputs/search_space_v2/threshold_f1_prevalence_guard/reports/中文Excel整理数据.json");
const outDir = path.join(projectRoot, "outputs/search_space_v2/threshold_f1_prevalence_guard/reports");
const data = JSON.parse(await fs.readFile(dataPath, "utf8"));

const outputFiles = {
  index_workbook: "当前指数对比数据_论文表格整理.xlsx",
  auc_f1_workbook: "AUC和F1都提升的组合.xlsx",
  auc_workbook: "AUC提升的组合.xlsx",
};

function colLetter(index) {
  let n = index + 1;
  let out = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

function safeSheetName(name) {
  return name.replace(/[\\/*?:[\]]/g, "_").slice(0, 31);
}

function matrixFromRecords(records) {
  if (!records || records.length === 0) return [["无数据"]];
  const headers = Object.keys(records[0]);
  return [headers, ...records.map((row) => headers.map((header) => row[header]))];
}

function setNumberFormats(sheet, matrix) {
  const headers = matrix[0];
  const rowCount = matrix.length;
  if (rowCount <= 1) return;
  for (let col = 0; col < headers.length; col += 1) {
    const header = String(headers[col]);
    if (
      header.includes("AUC") ||
      header.includes("F1") ||
      header.includes("准确率") ||
      header.includes("精确率") ||
      header.includes("召回率") ||
      header.includes("Accuracy") ||
      header.includes("比例") ||
      header.includes("阈值") ||
      header.includes("threshold") ||
      header.includes("gain") ||
      header.includes("rate") ||
      header.includes("precision") ||
      header.includes("recall")
    ) {
      sheet.getRangeByIndexes(1, col, rowCount - 1, 1).format.numberFormat = [["0.000000"]];
    }
  }
}

function writeSheet(workbook, name, records) {
  const sheet = workbook.worksheets.add(safeSheetName(name));
  sheet.showGridLines = false;
  const matrix = matrixFromRecords(records);
  const rows = matrix.length;
  const cols = matrix[0].length;
  const range = sheet.getRangeByIndexes(0, 0, rows, cols);
  range.values = matrix;
  range.format.font.name = "Microsoft YaHei";
  range.format.font.size = 10;
  range.format.wrapText = false;

  const header = sheet.getRangeByIndexes(0, 0, 1, cols);
  header.format.fill.color = "#1F4E79";
  header.format.font.color = "#FFFFFF";
  header.format.font.bold = true;
  header.format.rowHeightPx = 28;
  sheet.freezePanes.freezeRows(1);

  for (let col = 0; col < cols; col += 1) {
    const headerText = String(matrix[0][col] ?? "");
    let width = Math.max(90, Math.min(180, headerText.length * 12 + 36));
    if (headerText.includes("状态") || headerText.includes("status")) width = 270;
    if (headerText.includes("source")) width = 300;
    if (headerText.includes("方案")) width = 150;
    sheet.getRangeByIndexes(0, col, rows, 1).format.columnWidthPx = width;
  }
  setNumberFormats(sheet, matrix);
  if (rows > 1 && cols > 1) {
    const tableName = `${safeSheetName(name).replace(/[^A-Za-z0-9_]/g, "_")}_${Math.random().toString(36).slice(2, 7)}`;
    sheet.tables.add(`A1:${colLetter(cols - 1)}${rows}`, true, tableName);
  }
  return sheet;
}

async function buildWorkbook(kind, sheets) {
  const workbook = Workbook.create();
  for (const [sheetName, records] of Object.entries(sheets)) {
    writeSheet(workbook, sheetName, records);
  }
  const inspect = await workbook.inspect({
    kind: "sheet,table",
    include: "name,range",
    maxChars: 6000,
    tableMaxRows: 3,
    tableMaxCols: 6,
  });
  console.log(`--- ${outputFiles[kind]} ---`);
  console.log(inspect.ndjson);
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 20 },
    summary: "formula error scan",
  });
  console.log(errors.ndjson);
  await workbook.render({ sheetName: Object.keys(sheets)[0].slice(0, 31), range: "A1:H12", scale: 1, format: "png" });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  const outPath = path.join(outDir, outputFiles[kind]);
  await output.save(outPath);
  console.log(outPath);
}

await fs.mkdir(outDir, { recursive: true });
for (const [kind, sheets] of Object.entries(data)) {
  await buildWorkbook(kind, sheets);
}
