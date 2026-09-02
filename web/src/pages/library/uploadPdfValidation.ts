export async function isValidLibraryPdf(file: File): Promise<boolean> {
  const name = String(file.name || '').trim().toLowerCase()
  const mime = String(file.type || '').trim().toLowerCase()
  if (!name.endsWith('.pdf') && mime !== 'application/pdf') return false
  if (file.size < 5) return false
  try {
    const prefix = new Uint8Array(await file.slice(0, 1024).arrayBuffer())
    const signature = [0x25, 0x50, 0x44, 0x46, 0x2d]
    for (let offset = 0; offset <= prefix.length - signature.length; offset += 1) {
      if (signature.every((value, index) => prefix[offset + index] === value)) return true
    }
  } catch {
    return false
  }
  return false
}
