import type { Metadata } from "next";
import { Cormorant } from "next/font/google";
import { CircleCursor } from "@/app/_components/circle-cursor";
import cn from "classnames";
import "./globals.css";

const cormorant = Cormorant({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: `Brian Williams`,
  description: `Brian's Personal Website`,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          rel="apple-touch-icon"
          sizes="180x180"
          href="/favicon/apple-touch-icon.png"
        />
        <link
          rel="icon"
          type="image/png"
          sizes="32x32"
          href="/favicon/favicon-32x32.png"
        />
        <link
          rel="icon"
          type="image/png"
          sizes="16x16"
          href="/favicon/favicon-16x16.png"
        />
        <link rel="manifest" href="/favicon/site.webmanifest" />
        <link
          rel="mask-icon"
          href="/favicon/safari-pinned-tab.svg"
          color="#000000"
        />
        <link rel="shortcut icon" href="/favicon/favicon.ico" />
        <meta name="msapplication-TileColor" content="#000000" />
        <meta
          name="msapplication-config"
          content="/favicon/browserconfig.xml"
        />
        <meta name="theme-color" content="#000" />
        <link rel="alternate" type="application/rss+xml" href="/feed.xml" />
      </head>
      <body className={cn(cormorant.className)}>
        <div className="bg-gradient-to-b from-sky-100 via-zinc-100 to-rose-100">
          <div className="min-h-screen pb-8">{children}</div>
        </div>
        <CircleCursor />
      </body>
    </html>
  );
}
