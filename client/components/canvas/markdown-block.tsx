'use client';

import MarkdownIt from 'markdown-it';

const markdown = new MarkdownIt({ html: false, linkify: true, breaks: true });

export function MarkdownBlock({ content }: { content: string }) {
  return (
    <div
      className="markdown-block"
      dangerouslySetInnerHTML={{ __html: markdown.render(content) }}
    />
  );
}
