export const percent = (value: number | null, digits = 2) => value === null ? 'Unavailable' : `${(value * 100).toFixed(digits)}%`
export const points = (value: number | null) => value === null ? 'Unavailable' : `${value > 0 ? '+' : ''}${(value * 100).toFixed(2)} pp`
export const label = (value: string) => value.replaceAll('_', ' ')
export const decimal = (value: number | null, digits = 3) => value === null ? 'Unavailable' : value.toFixed(digits)
