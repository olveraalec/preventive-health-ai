/** Start only the pinned, independent, no-CSV Streamlit demonstration. */
import { execFileSync, spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { resolve, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const repo = fileURLToPath(new URL('../../', import.meta.url))
const worktree = process.env.EXPLORER_WORKTREE || resolve(repo, '..', 'hackathon-streamlit-demo')
const commit = 'c28e2ad51f2b9304f99637f6f720e41f2be92bdf'
if (!existsSync(worktree)) throw new Error('Create the separate Explorer worktree using frontend/STREAMLIT_RESEARCH_EXPLORER.md.')
if (execFileSync('git', ['-C', worktree, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim() !== commit) throw new Error('Explorer checkout is not the reviewed c28e2ad commit.')
for (const file of ['data/predictions.csv', 'data/eval_predictions.csv']) {
  if (existsSync(join(worktree, file))) throw new Error('This launcher requires the no-CSV synthetic demo. Use a separate, reviewed data workflow for real CSVs.')
}
const python = process.platform === 'win32' ? join(worktree, '.venv', 'Scripts', 'python.exe') : join(worktree, '.venv', 'bin', 'python')
if (!existsSync(python)) throw new Error('Install the Explorer-only hashed requirements in its separate .venv first.')
const child = spawn(python, ['-m', 'streamlit', 'run', 'dashboard/app.py',
  '--server.address', '127.0.0.1', '--server.port', '8501', '--server.headless', 'true',
  '--browser.gatherUsageStats', 'false'], {
  cwd: worktree, stdio: 'inherit', env: { ...process.env, PYTHON_DOTENV_DISABLED: '1' },
})
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal))
child.on('error', () => { console.error('Could not start the isolated Streamlit process.'); process.exitCode = 1 })
child.on('exit', code => { process.exitCode = code ?? 1 })
