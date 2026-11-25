import { posts } from '../lib/posts';
const escape = (value: any) => String(value).replace(/[<>&"']/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;','"':'&quot;',"'":'&apos;'}[c]!));
export function GET() {
  const base = 'https://oliver-zrz.github.io';
  return new Response(`<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>Runze Zhang · 开发手记</title><link>${base}</link><description>Runze Zhang 的开发、排障与环境配置笔记。</description><language>zh-CN</language>${posts.map(p => `<item><title>${escape(p.title)}</title><link>${base}${p.url}</link><guid>${base}${p.url}</guid><description>${escape(p.description)}</description><pubDate>${new Date(p.date).toUTCString()}</pubDate></item>`).join('')}</channel></rss>`, {headers: {'Content-Type':'application/rss+xml; charset=utf-8'}});
}
