import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium transition-[color,box-shadow] disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground elevation-button elevation-button-hover hover:bg-primary/90",
        destructive:
          "bg-destructive text-white elevation-button elevation-button-hover hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40",
        outline:
          "border border-input bg-background elevation-button elevation-button-hover hover:bg-accent hover:text-accent-foreground",
        secondary:
          "bg-secondary text-secondary-foreground elevation-button elevation-button-hover hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
        brand: "bg-[#2F6868] hover:bg-[#2F6868]/90 border-[#2F6868] text-white elevation-button elevation-button-hover",
      },
      size: {
        default: "h-9 px-4 py-2 has-[>svg]:px-3 [&_svg]:w-[var(--icon-size-button)] [&_svg]:h-[var(--icon-size-button)]",
        sm: "h-8 gap-1.5 px-3 has-[>svg]:px-2.5 [&_svg]:w-[var(--icon-size-sm)] [&_svg]:h-[var(--icon-size-sm)]",
        lg: "h-10 px-6 has-[>svg]:px-4 [&_svg]:w-[var(--icon-size-button)] [&_svg]:h-[var(--icon-size-button)]",
        icon: "size-9 [&_svg]:w-[var(--icon-size-button)] [&_svg]:h-[var(--icon-size-button)]",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

type ButtonProps = React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean;
  };

function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: ButtonProps) {
  const Comp = asChild ? Slot : "button";

  return (
    <Comp
      data-slot="button"
      className={cn(buttonVariants({ variant, size, className }))}
      style={{ borderRadius: "var(--radius-button)" }}
      {...props}
    />
  );
}

export { Button, buttonVariants, type ButtonProps };
