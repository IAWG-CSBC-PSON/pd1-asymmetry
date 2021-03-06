import numpy as np

DOWNLOAD_JS = r"""
function table_to_csv(source) {
    const columns = Object.keys(source.data)
    const nrows = source.get_length()
    const lines = [columns.join(',')]

    for (let i = 0; i < nrows; i++) {
        let row = [];
        for (let j = 0; j < columns.length; j++) {
            const column = columns[j]
            const entry = source.data[column][i].toString()
            const terminator = (entry.includes(',') ? '"' : '')
            row.push(terminator + entry + terminator)
        }
        lines.push(row.join(','))
    }
    return lines.join('\n').concat('\n')
}


const filename = 'data_result.csv'
const filetext = table_to_csv(source)
const blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' })

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename)
} else {
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.target = '_blank'
    link.style.visibility = 'hidden'
    link.dispatchEvent(new MouseEvent('click'))
}
"""


def round_signif(x, n=2):
    return float(
        np.format_float_positional(
            x, precision=n, unique=False, fractional=False, trim="k"
        )
    )
