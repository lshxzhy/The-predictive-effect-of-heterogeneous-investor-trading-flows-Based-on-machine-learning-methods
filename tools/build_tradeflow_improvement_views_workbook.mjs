import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = process.cwd();
const dataPath = path.join(projectRoot, "outputs/search_space_v2/reports/tradeflow_improvement_views_data.json");
const outPath = path.join(projectRoot, "outputs/search_space_v2/reports/tradeflow_improvement_views.xlsx");
const data = JSON.parse(await fs.readFile(dataPath, "utf8"));

const workbook = Workbook.create();

function colLetter(index) {
  let n = index + 1;
  let s = "";
  while (n > 0) {
    const r = (n - 1) % 26;
    s = String.fromCharCode(65 + r) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function matrixFromRecords(records) {
  if (!records || records.length === 0) return [["无数据"]];
  const headers = Object.keys(records[0]);
  const rows = records.map((record) => headers.map((header) => record[header]));
  return [headers, ...rows];
}

function writeSheet(sheetName, records, options = {}) {
  const sheet = workbook.worksheets.add(sheetName);
  sheet.showGridLines = false;
  const matrix = matrixFromRecords(records);
  const rowCount = matrix.length;
  const colCount = matrix[0].length;
  sheet.getRangeByIndexes(0, 0, rowCount, colCount).values = matrix;
  const used = sheet.getRangeByIndexes(0, 0, rowCount, colCount);
  used.format.font.name = "Microsoft YaHei";
  used.format.font.size = 10;
  used.format.wrapText = false;
  const header = sheet.getRangeByIndexes(0, 0, 1, colCount);
  header.format.fill.color = "#1F4E79";
  header.format.font.color = "#FFFFFF";
  header.format.font.bold = true;
  sheet.freezePanes.freezeRows(1);
  for (let col = 0; col < colCount; col += 1) {
    const headerText = String(matrix[0][col] ?? "");
    let width = Math.max(90, Math.min(210, headerText.length * 11 + 36));
    if (["finding", "best_examples"].includes(headerText)) width = headerText === "finding" ? 520 : 280;
    if (headerText.includes("gain") || headerText.includes("auc") || headerText.includes("f1") || headerText.includes("accuracy")) width = 105;
    sheet.getRangeByIndexes(0, col, rowCount, 1).format.columnWidthPx = width;
  }
  if (rowCount > 1 && colCount > 1) {
    const endCol = colLetter(colCount - 1);
    const table = sheet.tables.add(`A1:${endCol}${rowCount}`, true, `${sheetName.replace(/[^A-Za-z0-9_]/g, "_")}_tbl`);
    table.showFilterButton = true;
  }
  if (options.percentColumns) {
    for (const headerName of options.percentColumns) {
      const idx = matrix[0].indexOf(headerName);
      if (idx >= 0 && rowCount > 1) {
        sheet.getRangeByIndexes(1, idx, rowCount - 1, 1).format.numberFormat = [["0.00%"]];
      }
    }
  }
  return sheet;
}

writeSheet("观点总览", data.viewpoint);
writeSheet("正文推荐组合", data.recommended);
writeSheet("三指标同升", data.all3_up);
writeSheet("AUC_F1同升", data.both_auc_f1_up);
writeSheet("指数22日", data.index_22);
writeSheet("指数全部", data.index_all);
writeSheet("BU_22日", data.bu_22);
writeSheet("EG_22日", data.eg_22);
writeSheet("JM_22日", data.jm_22);
writeSheet("按模型", data.by_model, { percentColumns: ["auc_up_rate", "both_up_rate"] });
writeSheet("按期限", data.by_horizon, { percentColumns: ["auc_up_rate", "both_up_rate"] });
writeSheet("按资产", data.by_asset, { percentColumns: ["auc_up_rate", "both_up_rate"] });
writeSheet("模型x期限", data.by_model_horizon, { percentColumns: ["auc_up_rate", "both_up_rate"] });
writeSheet("资产x期限", data.by_asset_horizon, { percentColumns: ["auc_up_rate", "both_up_rate"] });
writeSheet("全部配对", data.all_pairwise);

const inspect = await workbook.inspect({
  kind: "sheet,table",
  include: "name,range",
  maxChars: 6000,
  tableMaxRows: 3,
  tableMaxCols: 6,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 20 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

await workbook.render({ sheetName: "观点总览", range: "A1:C6", scale: 1, format: "png" });
await workbook.render({ sheetName: "正文推荐组合", range: "A1:L12", scale: 1, format: "png" });

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outPath);
console.log(outPath);
