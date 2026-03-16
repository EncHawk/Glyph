declare module 'markdown-it' {
  export interface MarkdownItOptions {
    html?: boolean;
    linkify?: boolean;
    breaks?: boolean;
  }

  export default class MarkdownIt {
    constructor(options?: MarkdownItOptions);
    render(src: string): string;
  }
}
